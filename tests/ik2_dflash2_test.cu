#include "ik2_dflash2_kernels.cuh"
#include "ik2_draft_weights.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace df = ik2::draft::df2;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void gpu(cudaError_t err) { check(err == cudaSuccess, cudaGetErrorString(err)); }

template <class T> static T* upload(const std::vector<T>& v) {
    T* ptr;
    gpu(cudaMalloc(&ptr, v.size() * sizeof(T)));
    gpu(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

template <class T> static std::vector<T> read(T* ptr, size_t count) {
    std::vector<T> result(count);
    gpu(cudaMemcpy(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

static void weights_test() {
    using namespace ik2::draft;
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path() / ("ik2-weights-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(dir), "temporary directory");
    auto path = dir / "model.safetensors";
    std::string name = "candidate_selector.hidden_projection.weight";
    std::string header = "{\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[0,4]}}";
    uint64_t size = header.size();
    std::vector<uint16_t> expected{0x3f80, 0xbf80};
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out << header;
        out.write(reinterpret_cast<const char*>(expected.data()), 4);
    }
    for (auto& source : {path, dir}) {
        auto f = st_open(source.string());
        check(st_bf16(f, name, {2, 1}) == expected, "tensor data differs");
    }
    fs::resize_file(path, fs::file_size(path) - 1);
    bool rejected = false;
    try { st_open(path.string()); } catch (const std::runtime_error&) { rejected = true; }
    fs::remove_all(dir);
    check(rejected, "accepted truncated weights");
}

static void conv_test() {
    constexpr int rows = 8, w = df::width;
    std::vector<float> x(rows * w), dynamic(rows * df::dyn_width);
    std::vector<uint16_t> base(4 * w);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < w; ++c) x[r*w+c] = (r + c % 3) % 7;
        for (int j = 0; j < df::dyn_width; ++j) dynamic[r*df::dyn_width+j] = (r + j) % 5;
    }
    for (int j = 0; j < 4; ++j)
        std::fill_n(base.begin() + j*w, w, uint16_t(0x3f80 + j * 128)); // 1, 2, 4, 8
    auto dx = upload(x), dy = upload(x), dd = upload(dynamic);
    auto db = upload(base);
    for (int side = 0; side < 2; ++side) {
        df::conv<<<dim3(w / 256, rows), 256>>>(dx, dd, db, side, dy);
        auto y = read(dy, x.size());
        for (int r = 0; r < rows; ++r) for (int c = 0; c < w; ++c) {
            int j = r * df::dyn_width + side * 2 * df::groups + c / 16;
            float want = ((1 << (2 * side)) + dynamic[j]) * x[r*w+c];
            if (r) want += ((2 << (2 * side)) + dynamic[j+df::groups]) * x[(r-1)*w+c];
            check(y[r*w+c] == want, "two-tap convolution differs");
        }
    }
    gpu(cudaFree(dx)); gpu(cudaFree(dy)); gpu(cudaFree(dd)); gpu(cudaFree(db));
}

static void selector_test() {
    const int stride = df::codebook_rows, rows = 2;
    std::vector<float> base(rows * stride, 10000);
    for (int r = 0; r < rows; ++r) {
        auto first = base.begin() + r*stride;
        std::fill_n(first, df::vocab, -2);
        for (int i = 0; i < 32; ++i) first[(7919*i) % df::vocab] = 4;
        first[11] = 100; first[12] = 35;
    }
    auto logits = upload(base);
    auto parts = upload(std::vector<int>(rows * df::parts * df::top));
    auto ids = upload(std::vector<int>(rows * df::top));
    df::radix_parts<<<dim3(df::parts, rows), 256>>>(logits, stride, parts);
    df::merge_parts<<<rows, 128>>>(logits, stride, parts, ids);
    auto selected = read(ids, rows * df::top);
    std::vector<int> order(df::vocab);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return base[a] > base[b]; });
    for (int r = 0; r < rows; ++r)
        check(std::equal(order.begin(), order.begin()+df::top, selected.begin()+r*df::top), "top-16 differs");

    std::vector<uint16_t> pred(13 * df::rank), succ((*std::max_element(selected.begin(), selected.end())+1) * df::rank);
    std::vector<float> hidden(rows * df::rank, 1);
    pred[0] = 0x3f80;       // anchor 0: -40 for candidate 11. Post-correction cap selects 11 (60 > 35).
    pred[11*df::rank] = 0x4000;  // predecessor 11: -80, so row two selects 12.
    succ[11*df::rank] = 0xc220;
    auto dp = upload(pred), ds = upload(succ);
    auto dh = upload(hidden);
    auto tokens = upload(std::vector<int>(3));
    df::walk<<<1, 512>>>(logits, stride, ids, dh, dp, ds, rows, tokens);
    check(read(tokens, 3) == std::vector<int>({0,11,12}), "selector walk differs");
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    gpu(cudaStreamCreate(&stream));
    gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    df::radix_parts<<<dim3(df::parts, rows), 256, 0, stream>>>(logits, stride, parts);
    df::merge_parts<<<rows, 128, 0, stream>>>(logits, stride, parts, ids);
    df::walk<<<1, 512, 0, stream>>>(logits, stride, ids, dh, dp, ds, rows, tokens);
    gpu(cudaStreamEndCapture(stream, &graph));
    gpu(cudaGraphInstantiate(&exec, graph, 0));
    for (int anchor : {0, 11}) {
        gpu(cudaMemcpy(tokens, &anchor, sizeof(anchor), cudaMemcpyHostToDevice));
        gpu(cudaGraphLaunch(exec, stream));
        gpu(cudaStreamSynchronize(stream));
        const auto want = anchor == 0 ? std::vector<int>{0, 11, 12} : std::vector<int>{11, 12, 11};
        check(read(tokens, 3) == want, "selector graph replay differs");
    }
    gpu(cudaGraphExecDestroy(exec));
    gpu(cudaGraphDestroy(graph));
    gpu(cudaStreamDestroy(stream));
    gpu(cudaFree(logits)); gpu(cudaFree(parts)); gpu(cudaFree(ids));
    gpu(cudaFree(dp)); gpu(cudaFree(ds)); gpu(cudaFree(dh)); gpu(cudaFree(tokens));
}

int main(int argc, char** argv) {
    try {
        weights_test();
        if (argc == 2 && std::strcmp(argv[1], "--weights-only") == 0) return 0;
        conv_test();
        selector_test();
        std::puts("DFlash2 weights, convolution, top-16 and selector passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
