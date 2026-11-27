#include "ik2_draft.hpp"
#include <random>
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

#ifndef STRATA_USE_HIP
static void projection_test() {
    constexpr int n = 32, k = 256, stride = n + 7;
    std::mt19937 rng(41);
    std::vector<uint16_t> w(n*k), x(64*k);
    auto sign = [&](uint16_t& v) { v = rng() & 1 ? 0x3f80 : 0xbf80; };
    std::for_each(w.begin(), w.end(), sign);
    std::for_each(x.begin(), x.end(), sign);
    auto* dw = upload(w);
    auto* dx = upload(x);
    auto* packed = upload(w);
    auto* dy = upload(std::vector<float>(64 * stride));
    df::pack_projection<<<dim3(n/16, k/64), 128>>>(dw, k, reinterpret_cast<uint4*>(packed));
    for (int rows : {1, 7, 13, 26, 49, 64}) {
        df::project_rows(packed, dx, n, k, rows, dy, stride, nullptr);
        const auto y = read(dy, rows * stride);
        for (int t = 0; t < rows; ++t) for (int c = 0; c < n; ++c) {
            int expected = 0;
            for (int j = 0; j < k; ++j) expected += w[c*k+j] == x[t*k+j] ? 1 : -1;
            check(y[t*stride+c] == expected, "packed BF16 projection differs from the dot product");
        }
    }
    gpu(cudaFree(dw)); gpu(cudaFree(dx)); gpu(cudaFree(packed)); gpu(cudaFree(dy));
}

static void proposals_test(const char* checkpoint) {
    using namespace ik2::draft;
    constexpr int slots = 8, C = df::width, ld = slots * kBlock;
    cudaStream_t stream;
    cublasHandle_t blas;
    gpu(cudaStreamCreate(&stream));
    check(cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS, "create BLAS handle");
    check(cublasSetStream(blas, stream) == CUBLAS_STATUS_SUCCESS, "set BLAS stream");
    auto* batch = load(checkpoint, C, 32, blas, stream, slots);
    auto* reference = load(checkpoint, C, 32, blas, stream);
    std::mt19937 rng(217);
    std::vector<int8_t> q(size_t(df::vocab) * C);
    for (auto& x : q) x = int(rng() % 255) - 127;
    auto* head = upload(q);
    auto* scales = upload(std::vector<uint16_t>(q.size() / 32, 0x2000));
    std::vector<float> data(4 * ld * C);
    for (auto& x : data) x = (int(rng() % 8192) - 4096) / 1024.f;
    auto* taps = upload(data);
    for (auto* d : {batch, reference}) {
        check(dflash2_mask(d) != nullptr, "expected a DFlash2 checkpoint");
        gpu(cudaMemcpy(dflash2_mask(d), data.data(), C * sizeof(float), cudaMemcpyHostToDevice));
        set_head_q8(d, head, scales);
    }
    struct Step {
        int window;
        std::vector<ContextWindow> writes;
        std::vector<int> ids, positions, anchors, expected;
    };
    std::vector<Step> steps(15);
    int lengths[slots]{};
    std::vector<int> order{3, 7, 1, 5, 0, 2, 6, 4};
    for (int n = 0; n < int(steps.size()); ++n) {
        auto& step = steps[n];
        const int active = 2 + n % 7;
        step.window = 2 + (n * 3) % 7;
        std::rotate(order.begin(), order.begin() + 1, order.end());
        step.ids.assign(order.begin(), order.begin() + active);
        for (int i = 0; i < active; ++i) {
            const int slot = order[i], count = 1 + (i + n) % kBlock;
            if (n == 8 && i == 0) lengths[slot] = 0;
            step.writes.push_back({slot, i * kBlock, count, lengths[slot]});
            step.positions.push_back(lengths[slot] += count);
            step.anchors.push_back(rng() % df::vocab);
        }
        step.expected.resize(active * step.window);
    }
    // Replay an entire request history before reusing the one-slot reference model.
    for (int slot = 0; slot < slots; ++slot) {
        for (int n = 0; n < int(steps.size()); ++n) {
            auto& step = steps[n];
            for (int i = 0; i < int(step.ids.size()); ++i) {
                const auto& w = step.writes[i];
                if (w.slot != slot) continue;
                write_context(reference, taps, ld, w.first, w.count, w.position);
                int* out = step.expected.data() + i * step.window;
                out[0] = step.anchors[i];
                propose(reference, out[0], taps + (n + i + 1) * C, step.positions[i], nullptr,
                        df::vocab, df::vocab, out + 1, step.window - 1);
            }
        }
    }
    // A stable input address exercises graph replay with fresh slot IDs, positions and anchors.
    auto* embeddings = upload(std::vector<float>(slots * C));
    for (int n = 0; n < int(steps.size()); ++n) {
        const auto& step = steps[n];
        const int active = step.ids.size();
        std::vector<int> actual(active * step.window, -1);
        for (int i = 0; i < active; ++i) actual[i * step.window] = step.anchors[i];
        gpu(cudaMemcpyAsync(embeddings, taps + (n + 1) * C, size_t(active) * C * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
        commit_windows(batch, taps, ld, step.writes);
        propose_batch(batch, active, step.ids.data(), step.positions.data(), embeddings, nullptr,
                      df::vocab, df::vocab, step.window, actual.data());
        for (int j = 0; j < int(actual.size()); ++j) {
            if (actual[j] == step.expected[j]) continue;
            const int i = j / step.window;
            std::fprintf(stderr, "step %d slot %d position %d draft %d: got %d expected %d\n",
                         n, step.ids[i], step.positions[i], j % step.window, actual[j], step.expected[j]);
            check(false, "batched draft IDs differ from independent proposals");
        }
    }
    std::puts("Batched proposals matched independent slots across ring wrap and reuse");
}
#endif

int main(int argc, char** argv) {
    try {
        weights_test();
        if (argc == 2 && std::strcmp(argv[1], "--weights-only") == 0) return 0;
#ifndef STRATA_USE_HIP
        projection_test();
        if (argc == 3 && std::strcmp(argv[1], "--checkpoint") == 0) {
            proposals_test(argv[2]);
            return 0;
        }
#endif
        conv_test();
        selector_test();
        std::puts("DFlash2 weights, convolution, top-16 and selector passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
