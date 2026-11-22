#include "ik2_kernels.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace ik2::k {
namespace {

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

__device__ uint32_t iq4_values(uint32_t codes) {
    const uint32_t p = codes & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(0xf6eaddcfu, 0xbfad9881u, p);
    const uint32_t hi = __builtin_amdgcn_perm(0x71594535u, 0x26190d01u, p);
    return __builtin_amdgcn_perm(hi, lo, ((codes >> 1) & 0x04040404u) | 0x03020100u);
}

// Each lane owns a consecutive group of sixteen weights in an output row.
template <int Type>
__device__ Half16 expert_fragment(const uint8_t* block, int half) {
    Half16 out;
    const float scale = __half2float(__ushort_as_half(*(const uint16_t*)block));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t packed;
        if constexpr (Type == kIQ4_NL) {
            __builtin_memcpy(&packed, block + 2 + i * 4, 4);
            packed = iq4_values(packed >> (half * 4));
        } else {
            __builtin_memcpy(&packed, block + 2 + half * 16 + i * 4, 4);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[i * 4 + j] = (_Float16)(scale * (int8_t)(packed >> (j * 8)));
    }
    return out;
}

template <int Type>
__global__ void expert_wmma_gemm(const uint8_t* const* experts, size_t offset, const int* starts,
                                  const uint16_t* input, int K, int N, float* output) {
    constexpr int block_bytes = Type == kIQ4_NL ? 18 : 34;
    const int expert = blockIdx.y;
    const int first = starts[expert] + blockIdx.z * 16, end = starts[expert + 1];
    if (first >= end) return;
    const int lane = threadIdx.x & 31, lane16 = lane & 15;
    const int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + lane16;
    if (col - lane16 >= N) return;
    const int token = first + lane16;
    const uint8_t* w = experts[expert] + offset + (size_t)col * (K / 32) * block_bytes;
    Float8 result = {};
    for (int k = 0; k < K; k += 32) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            Half16 a = {};
            if (token < end)
                __builtin_memcpy(&a, input + (size_t)token * K + k + half * 16, sizeof(a));
            const Half16 b = expert_fragment<Type>(w + (k / 32) * block_bytes, half);
            result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, result);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = first + 2 * i + (lane >> 4);
        if (row < end) output[(size_t)row * N + col] = result[i];
    }
}

}  // namespace

void rdna3_expert_gemm(int type, const uint8_t* const* experts, size_t offset, int count, const int* starts,
                       int max_rows, const uint16_t* input, int K, int N, float* output, cudaStream_t stream) {
    const dim3 grid((N + 63) / 64, count, (max_rows + 15) / 16);
    if (type == kIQ4_NL)
        expert_wmma_gemm<kIQ4_NL><<<grid, 128, 0, stream>>>(experts, offset, starts, input, K, N, output);
    else if (type == kQ8_0)
        expert_wmma_gemm<kQ8_0><<<grid, 128, 0, stream>>>(experts, offset, starts, input, K, N, output);
    else {
        std::fprintf(stderr, "rdna3_expert_gemm: unsupported expert type %d\n", type);
        std::abort();
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "rdna3_expert_gemm: %s\n", hipGetErrorString(status));
        std::abort();
    }
}
}  // namespace ik2::k
