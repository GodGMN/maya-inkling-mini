// Input from ik2_host_ref_fixture.py. Both engine entry points run on the CPU.
#include "ik2_cpu.hpp"
#include "ik2_kernels.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <class T>
std::vector<T> read_values(size_t count) {
    std::vector<T> values(count);
    if (std::fread(values.data(), sizeof(T), count, stdin) != count) {
        std::fprintf(stderr, "short fixture; pipe ik2_host_ref_fixture.py into this test\n");
        std::exit(1);
    }
    return values;
}

int main() {
    constexpr int C = 128, F = 64;
    std::array<std::vector<uint8_t>, 3> weights;
    for (auto& matrix : weights) {
        matrix = read_values<uint8_t>(C * F / 64 * 36);
        const auto expected = read_values<float>(C * F);
        std::vector<float> decoded(expected.size());
        ik2::k::decode_host(ik2::k::kNVFP4, matrix.data(), decoded.size(), decoded.data());
        if (decoded != expected) {
            std::fprintf(stderr, "NVFP4 host decode differs from the fixture\n");
            return 1;
        }
    }
    auto x = read_values<float>(C);
    auto expected = read_values<float>(C);
    std::vector<float> y(C);
    ik2::cpu::Pool pool(2);
    ik2::cpu::ExpertJob job;
    job.gate = weights[0].data();
    job.up = weights[1].data();
    job.down = weights[2].data();
    job.sg = 0.75f;
    job.su = 1.25f;
    job.x = x.data();
    job.y = y.data();
    ik2::cpu::post_expert(pool, &job, C, F, 4.f, 25.f);
    pool.wait();
    float max_error = 0;
    for (int i = 0; i < C; ++i) {
        const float error = std::abs(y[i] - expected[i]);
        if (!(error <= 2e-5f * (1.f + std::abs(expected[i])))) {
            std::fprintf(stderr, "output[%d] = %g, reference %g\n", i, y[i], expected[i]);
            return 1;
        }
        max_error = std::max(max_error, error);
    }
    std::printf("NVFP4 decode exact; Ref.moe max absolute error %.9g\n", max_error);
}
