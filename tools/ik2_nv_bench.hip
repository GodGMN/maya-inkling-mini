// Time the public single-token expert operations on generated GGUF blocks.
#include "ik2_kernels.cuh"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace k = ik2::k;

static void check(hipError_t result) {
    if (result == hipSuccess) return;
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
}

template <typename T>
static T* allocate(size_t count) {
    T* ptr = nullptr;
    check(hipMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
static T* upload(const std::vector<T>& values) {
    auto* ptr = allocate<T>(values.size());
    check(hipMemcpy(ptr, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    return ptr;
}

int main() {
    constexpr int width = 2304, hidden_width = 1152, active = 8, experts = 128, steps = 96;
    constexpr size_t matrix_bytes = size_t(width) * hidden_width / 64 * 36;
    std::mt19937 rng(521);
    std::vector<uint8_t> expert(3 * matrix_bytes);
    for (auto& value : expert) value = static_cast<uint8_t>(rng());
    for (size_t block = 0; block < expert.size(); block += 36)
        for (int s = 0; s < 4; ++s) expert[block + s] = 16 + rng() % 17;

    auto* weights = allocate<uint8_t>(expert.size() * experts);
    for (int i = 0; i < experts; ++i)
        check(hipMemcpy(weights + i * expert.size(), expert.data(), expert.size(), hipMemcpyHostToDevice));
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::vector<const uint8_t*> schedule;
    for (int step = 0; step < steps; ++step) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < active; ++i) schedule.push_back(weights + order[i] * expert.size());
    }
    auto** picks = upload(schedule);
    std::vector<float> x(width);
    for (auto& value : x) value = (int(rng() % 2001) - 1000) / 1000.f;
    auto* input = upload(x);
    auto* token = upload(std::vector<int>(active, 0));
    auto* limbs = allocate<uint16_t>(2 * width);
    auto* hidden = allocate<uint16_t>(active * hidden_width);
    auto* output = allocate<float>(active * width);
    k::to_f16_limbs(input, width, limbs, nullptr);

    hipEvent_t begin, end;
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
    std::puts("projection,method,repeat,microseconds");
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (bool down : {false, true}) {
            for (int method = 0; method < 2; ++method) {
                const bool wmma = (method + repeat) % 2;
                auto launch = [&](int step) {
                    auto** selected = picks + step * active;
                    if (down && wmma)
                        k::experts_down_mma(selected, active, 2 * matrix_bytes, hidden, width, hidden_width, output, nullptr);
                    else if (down)
                        k::experts_down(selected, active, k::kNVFP4, 2 * matrix_bytes, hidden, width, hidden_width, output, nullptr);
                    else if (wmma)
                        k::experts_gate_up_swiglu_mma(selected, active, matrix_bytes, limbs, width, hidden_width,
                                                     nullptr, nullptr, 20.f, 20.f, hidden, nullptr);
                    else
                        k::experts_gate_up_swiglu(selected, token, active, k::kNVFP4, k::kNVFP4, matrix_bytes, input,
                                                 width, hidden_width, nullptr, nullptr, 20.f, 20.f, hidden, nullptr);
                };
                for (int step = 0; step < 12; ++step) launch(step);
                check(hipEventRecord(begin));
                for (int step = 0; step < steps; ++step) launch(step);
                check(hipEventRecord(end));
                check(hipEventSynchronize(end));
                float ms = 0;
                check(hipEventElapsedTime(&ms, begin, end));
                std::printf("%s,%s,%d,%.3f\n", down ? "down" : "gate_up", wmma ? "wmma" : "scalar", repeat + 1, ms * 1000 / steps);
            }
        }
    }
    check(hipEventDestroy(begin));
    check(hipEventDestroy(end));
    for (void* ptr : std::initializer_list<void*>{output, hidden, limbs, token, input, picks, weights}) check(hipFree(ptr));
}
