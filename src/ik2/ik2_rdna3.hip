#include "ik2_kernels.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace ik2::k {
namespace {

void check_launch(const char* kernel) {
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", kernel, hipGetErrorString(status));
        std::abort();
    }
}

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
constexpr int kDenseF16 = 1;
constexpr int kPlanarQ8 = -1;

__device__ uint32_t iq4_values(uint32_t codes) {
    const uint32_t p = codes & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(0xf6eaddcfu, 0xbfad9881u, p);
    const uint32_t hi = __builtin_amdgcn_perm(0x71594535u, 0x26190d01u, p);
    return __builtin_amdgcn_perm(hi, lo, ((codes >> 1) & 0x04040404u) | 0x03020100u);
}

using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using HalfWords = uint32_t __attribute__((ext_vector_type(8)));

// FP16 encodes 1024..1279 at unit spacing. Bias two signed bytes into this range, subtract 1152 exactly,
// then multiply by the block's FP16 scale. Each packed multiply rounds once, as in dequantize-to-half.
template <bool Upper>
__device__ __forceinline__ uint32_t scaled_byte_pair(uint32_t bytes, _Float16 scale) {
    const uint32_t biased = __builtin_amdgcn_perm(0x64646464u, bytes ^ 0x80808080u, Upper ? 0x04030402u : 0x04010400u);
    const Half2 values = __builtin_bit_cast(Half2, biased) - (_Float16)1152;
    return __builtin_bit_cast(uint32_t, values * scale);
}

__device__ __forceinline__ Half16 planar_fragment(const int8_t* weights, _Float16 scale) {
    using Words = uint32_t __attribute__((ext_vector_type(4)));
    Words words;
    HalfWords halves;
    __builtin_memcpy(&words, weights, sizeof(words));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        halves[i * 2] = scaled_byte_pair<false>(words[i], scale);
        halves[i * 2 + 1] = scaled_byte_pair<true>(words[i], scale);
    }
    return __builtin_bit_cast(Half16, halves);
}

// GGUF stores four unsigned E4M3 scales and 32 packed E2M1 bytes per 64 weights.
__device__ _Float16 nv_scale(uint8_t x) {
    const unsigned e = (x >> 3) & 15, m = x & 7;
    if (e == 0) return (_Float16)m * (_Float16)0x1p-10f;
    return x == 0x7f ? (_Float16)0 : __builtin_bit_cast(_Float16, uint16_t(((e + 7) << 10) | (m << 7)));
}

// Each E2M1 value times two fits in the high byte of an IEEE half. nv_scale supplies the other factor of 1/2.
__device__ __forceinline__ uint2 nv_four_halves(uint32_t codes, _Float16 scale) {
    const uint32_t magnitude = __builtin_amdgcn_perm(0x4a484644u, 0x42403c00u, codes & 0x07070707u);
    const uint32_t high = magnitude | ((codes & 0x08080808u) << 4);
    const Half2 lo = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0u, high, 0x01040004u));
    const Half2 hi = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0u, high, 0x03040204u));
    return make_uint2(__builtin_bit_cast(uint32_t, lo * scale), __builtin_bit_cast(uint32_t, hi * scale));
}

template <int Type>
__device__ size_t expert_byte_offset(size_t value) {
    if constexpr (Type == kNVFP4) return (value / 64) * 36;
    return (value / 32) * (Type == kIQ4_NL ? 18 : 34);
}

// Each lane owns a consecutive group of sixteen weights in an output row.
template <int Type>
__device__ Half16 expert_fragment(const uint8_t* blocks, size_t first) {
    const uint8_t* block = blocks + expert_byte_offset<Type>(first);
    const int sub = (first / 16) % (Type == kNVFP4 ? 4 : 2);
    _Float16 scale;
    if constexpr (Type == kNVFP4) scale = nv_scale(block[sub]);
    else scale = __builtin_bit_cast(_Float16, *(const uint16_t*)block);
    HalfWords out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t packed;
        if constexpr (Type == kNVFP4) {
            __builtin_memcpy(&packed, block + 4 + sub * 8 + (i % 2) * 4, 4);
            const uint2 values = nv_four_halves(packed >> ((i / 2) * 4), scale);
            out[i * 2] = values.x;
            out[i * 2 + 1] = values.y;
            continue;
        } else if constexpr (Type == kIQ4_NL) {
            __builtin_memcpy(&packed, block + 2 + i * 4, 4);
            packed = iq4_values(packed >> (sub * 4));
        } else {
            __builtin_memcpy(&packed, block + 2 + sub * 16 + i * 4, 4);
        }
        out[i * 2] = scaled_byte_pair<false>(packed, scale);
        out[i * 2 + 1] = scaled_byte_pair<true>(packed, scale);
    }
    return __builtin_bit_cast(Half16, out);
}

template <int Type>
__device__ __forceinline__ Float8 matrix_product(Half16 a, Half16 b, Float8 sum) {
    // Accumulate independently rounded 16-channel dots when NVFP4 sub-block scales span many exponents.
    if constexpr (Type == kNVFP4)
        return sum + __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, Float8{});
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sum);
}

template <bool Gate> constexpr int nv_token_waves = Gate ? 16 : 4;

// Sixteen output channels share one input. Two WMMA rows carry its high and low FP16 limbs.
template <bool Gate>
__global__ void nv_token_wmma(const uint8_t* const* experts, size_t offset, const uint16_t* input,
                              int K, int N, const float* sg, const float* su, float gate_cap,
                              float up_cap, uint16_t* hidden, float* output) {
    constexpr int splits = nv_token_waves<Gate>;
    __shared__ float partial[splits][Gate ? 2 : 1][16];
    const int lane = threadIdx.x & 31, col = lane & 15, split = threadIdx.x / 32;
    const int slot = blockIdx.y, row = blockIdx.x * 16 + col;
    const uint8_t* w = experts[slot] + expert_byte_offset<kNVFP4>((size_t)row * K) + (Gate ? 0 : offset);
    const uint16_t* x = input + (Gate ? (lane & 1) * K : (size_t)slot * K);
    Float8 first = {}, second = {};
    for (int k = split * 16; k < K; k += splits * 16) {
        Half16 a;
        __builtin_memcpy(&a, x + k, sizeof(a));
        first = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, expert_fragment<kNVFP4>(w, k), first);
        if constexpr (Gate)
            second = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, expert_fragment<kNVFP4>(w + offset, k), second);
    }
    if constexpr (Gate) {
        const float g = first[0] + __shfl_xor(first[0], 16);
        const float u = second[0] + __shfl_xor(second[0], 16);
        if (lane < 16) {
            partial[split][0][col] = g;
            partial[split][1][col] = u;
        }
    } else if (lane < 16) partial[split][0][col] = first[0];
    __syncthreads();
    if (threadIdx.x >= 16) return;
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int i = 0; i < splits; ++i) {
        g += partial[i][0][col];
        if constexpr (Gate) u += partial[i][1][col];
    }
    if constexpr (Gate) {
        g *= sg ? sg[slot] : 1.f;
        u *= su ? su[slot] : 1.f;
        const float v = gate_cap * tanhf(g / gate_cap) * (1.f / (1.f + expf(-g))) * up_cap * tanhf(u / up_cap);
        hidden[(size_t)slot * N + row] = __builtin_bit_cast(uint16_t, (_Float16)v);
    } else output[(size_t)slot * N + row] = g;
}

// One workgroup assigns each routed token a row in its expert matrix. Its row within the expert can vary,
// but the inverse permutation always sums the selected experts in increasing expert ID.
__global__ void group_expert_rows(const int* top, const float* weights, int slots, int K, int experts,
                                  const float* gs, const float* us, const float* ds, ExpertTiles tiles,
                                  int* tokens, int* inv, float* gate, float* up, float* weight) {
    extern __shared__ int counts[];
    int* starts = counts + experts;
    int* tile_starts = starts + experts;
    const int tid = threadIdx.x;
    for (int e = tid; e < experts; e += blockDim.x) counts[e] = 0;
    for (int i = tid; i < tiles.capacity; i += blockDim.x) tiles.data[i] = {0, 0, 0};
    __syncthreads();
    for (int i = tid; i < slots; i += blockDim.x) atomicAdd(counts + top[i], 1);
    __syncthreads();
    if (tid == 0) {
        int rows = 0, ntiles = 0;
        for (int e = 0; e < experts; ++e) {
            starts[e] = rows;
            tile_starts[e] = ntiles;
            rows += counts[e];
            ntiles += (counts[e] + tiles.rows - 1) / tiles.rows;
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        const int end = starts[e] + counts[e];
        for (int row = starts[e], idx = tile_starts[e]; row < end; row += tiles.rows, ++idx)
            tiles.data[idx] = {e, row, min(end, row + tiles.rows)};
        counts[e] = starts[e];
    }
    __syncthreads();
    for (int i = tid; i < slots; i += blockDim.x) {
        const int e = top[i], token = i / K;
        const int row = atomicAdd(counts + e, 1);
        tokens[row] = token;
        gate[row] = gs ? gs[e] : 1.f;
        up[row] = us ? us[e] : 1.f;
        weight[row] = weights[i] * (ds ? ds[e] : 1.f);
        int rank = 0;
        for (int j = 0; j < K; ++j) rank += top[token * K + j] < e;
        inv[token * K + rank] = row;
    }
}

template <int Type, int Rows>
__global__ void expert_wmma_gemm(const uint8_t* const* experts, size_t offset, const ExpertTile* tiles,
                                  const uint16_t* input, int K, int N, float* output) {
    const auto [expert, first, end] = tiles[blockIdx.y];
    if (first >= end) return;
    const int lane = threadIdx.x & 31, lane16 = lane & 15;
    const int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + lane16;
    if (col - lane16 >= N) return;
    const int token = first + lane16;
    const uint8_t* w = experts[expert] + offset + expert_byte_offset<Type>((size_t)col * K);
    Float8 result[Rows / 16] = {};
    for (int k = 0; k < K; k += 32) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const Half16 b = expert_fragment<Type>(w, k + half * 16);
#pragma unroll
            for (int m = 0; m < Rows / 16; ++m) {
                if (m && first + m * 16 >= end) continue;
                Half16 a = {};
                if (token + m * 16 < end)
                    __builtin_memcpy(&a, input + (size_t)(token + m * 16) * K + k + half * 16, sizeof(a));
                result[m] = matrix_product<Type>(a, b, result[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Rows / 16; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = first + m * 16 + 2 * i + (lane >> 4);
            if (row < end) output[(size_t)row * N + col] = result[m][i];
        }
    }
}

// A workgroup computes 64 tokens by 64 output channels. Each wave owns a 32x32 quadrant.
// The extra eight halfs in each LDS row shift consecutive rows to different banks.
struct ExpertActivation {
    const float *gate_scale, *up_scale;
    float gate_cap, up_cap;
    uint16_t* hidden;
};

template <int Type, int Depth, bool Activate = false>
__global__ void tiled_gemm(const uint16_t* dense, const uint8_t* const* experts, size_t offset, const ExpertTile* tiles,
                           const uint16_t* input, int N, int K, int T, float* output, int ld,
                           const int8_t* planar = nullptr, const uint16_t* scales = nullptr, ExpertActivation activation = {}) {
    constexpr int vectors = Depth / 8, pitch = vectors + 1;
    __shared__ uint4 a[64][pitch], b[64][pitch];
    const int tid = threadIdx.x, lane = tid & 31, l = lane & 15, wave = tid / 32;
    int row0 = blockIdx.y * 64;
    const uint8_t* quant = nullptr;
    if constexpr (Type == kIQ4_NL || Type == kQ8_0 || Type == kNVFP4) {
        const auto tile = tiles[blockIdx.y];
        if (tile.first >= tile.end) return;
        row0 = tile.first;
        T = tile.end;
        quant = experts[tile.expert] + offset;
    }
    const int col0 = blockIdx.x * (Activate ? 32 : 64);
    const int m0 = (wave / 2) * 32, n0 = (wave % 2) * 32;
    Float8 accum[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += Depth) {
#pragma unroll
        for (int load = 0; load < 64 * vectors / 128; ++load) {
            const int f = tid + load * 128, row = f / vectors, vec = f % vectors;
            const int k = k0 + vec * 8;
            uint4 av = {};
            if (row0 + row < T) __builtin_memcpy(&av, input + (size_t)(row0 + row) * K + k, sizeof(av));
            a[row][vec] = av;
            if constexpr (Type == kDenseF16) {
                uint4 bv = {};
                if (col0 + row < N) __builtin_memcpy(&bv, dense + (size_t)(col0 + row) * K + k, sizeof(bv));
                b[row][vec] = bv;
            }
        }
        if constexpr (Type != kDenseF16) {
#pragma unroll
            for (int f = tid; f < 64 * (Depth / 16); f += 128) {
                const int row = f / (Depth / 16), frag = f % (Depth / 16);
                Half16 bv = {};
                const int channel = col0 + (Activate ? (row / 32) * 16 + row % 16 : row);
                if (channel < N) {
                    const int weight_row = channel + (Activate && (row & 16) ? N : 0);
                    const size_t base = (size_t)weight_row * K + k0 + frag * 16;
                    if constexpr (Type == kPlanarQ8) {
                        const _Float16 scale = __builtin_bit_cast(_Float16, scales[base / 32]);
                        bv = planar_fragment(planar + base, scale);
                    } else {
                        bv = expert_fragment<Type>(quant, base);
                    }
                }
                __builtin_memcpy(&b[row][2 * frag], &bv, sizeof(bv));
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < vectors; kk += 2) {
            Half16 ax[2], bx[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                __builtin_memcpy(&ax[j], &a[m0 + j * 16 + l][kk], sizeof(Half16));
                __builtin_memcpy(&bx[j], &b[n0 + j * 16 + l][kk], sizeof(Half16));
            }
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int n = 0; n < 2; ++n)
                    accum[m][n] = matrix_product<Type>(ax[m], bx[n], accum[m][n]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + m0 + m * 16 + 2 * i + (lane >> 4);
            if (row >= T) continue;
            if constexpr (Activate) {
                const int col = col0 + (wave % 2) * 16 + l;
                if (col < N) {
                    const float g = accum[m][0][i] * activation.gate_scale[row];
                    const float u = accum[m][1][i] * activation.up_scale[row];
                    const float value = activation.gate_cap * tanhf(g / activation.gate_cap) * (1.f / (1.f + expf(-g))) *
                                        activation.up_cap * tanhf(u / activation.up_cap);
                    activation.hidden[(size_t)row * N + col] = __builtin_bit_cast(uint16_t, (_Float16)value);
                }
            } else {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int col = col0 + n0 + n * 16 + l;
                    if (col < N) output[(size_t)row * ld + col] = accum[m][n][i];
                }
            }
        }
    }
}

template <int Type, bool Activate = false>
void launch_tiled(dim3 grid, const uint16_t* dense, const uint8_t* const* experts, size_t offset,
                  const ExpertTile* tiles, const uint16_t* x, int N, int K, int T, float* y, int ld,
                  cudaStream_t stream, const int8_t* planar = nullptr, const uint16_t* scales = nullptr, ExpertActivation activation = {}) {
    static const bool wide = getenv("IK2_WMMA_K") && atoi(getenv("IK2_WMMA_K")) == 64;
    if (wide && K % 64 == 0)
        tiled_gemm<Type, 64, Activate><<<grid, 128, 0, stream>>>(dense, experts, offset, tiles, x, N, K, T, y, ld, planar, scales, activation);
    else
        tiled_gemm<Type, 32, Activate><<<grid, 128, 0, stream>>>(dense, experts, offset, tiles, x, N, K, T, y, ld, planar, scales, activation);
}

__global__ void dense_q8_wmma(const int8_t* weights, const uint16_t* scales, int N, int K,
                              const uint16_t* input, int T, float* output, int ld) {
    const int lane = threadIdx.x & 31, l = lane & 15;
    const int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + l;
    const int base = blockIdx.y * 16, token = base + l;
    Float8 result = {};
    for (int k = 0; k < K; k += 32) {
        const _Float16 scale = col < N ? __builtin_bit_cast(_Float16, scales[((size_t)col * K + k) / 32]) : (_Float16)0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            Half16 a = {}, b = {};
            if (token < T) __builtin_memcpy(&a, input + (size_t)token * K + k + half * 16, sizeof(a));
            if (col < N) {
                b = planar_fragment(weights + (size_t)col * K + k + half * 16, scale);
            }
            result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, result);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = base + 2 * i + (lane >> 4);
        if (row < T && col < N) output[(size_t)row * ld + col] = result[i];
    }
}

__device__ __forceinline__ float wsum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

__device__ __forceinline__ float nv_ue4m3_f32(uint8_t x) {
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    const float norm_f = __uint_as_float(((e + 119u) << 23) | (m << 20));
    const float sub_f = (float)m * 0x1.0p-10f;
    const float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}

__device__ __forceinline__ float2 h2u_to_f2(uint32_t u) {
    const Half2 h = __builtin_bit_cast(Half2, u);
    return make_float2((float)h[0], (float)h[1]);
}

__device__ __forceinline__ float fdot2_u32(uint32_t a, uint32_t b, float acc) {
#if __has_builtin(__builtin_amdgcn_fdot2)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(Half2, a), __builtin_bit_cast(Half2, b), acc, false);
#else
    const float2 fa = h2u_to_f2(a), fb = h2u_to_f2(b);
    return fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, acc));
#endif
}

// Four E2M1 nibbles (in the low nibbles of each byte of `codes`) -> four FP32 weights (scaled by 2x,
// matched by nv_ue4m3_f32's 0.5x bias).
__device__ __forceinline__ float4 nv4_f32(uint32_t codes) {
    const uint32_t mag = __builtin_amdgcn_perm(0x4a484644u, 0x42403c00u, codes & 0x07070707u);
    const uint32_t hi = mag | ((codes & 0x08080808u) << 4);
    const float2 lo = h2u_to_f2(__builtin_amdgcn_perm(0u, hi, 0x01040004u));
    const float2 h2 = h2u_to_f2(__builtin_amdgcn_perm(0u, hi, 0x03040204u));
    return make_float4(lo.x, lo.y, h2.x, h2.y);
}

__device__ __forceinline__ float dot4_f4(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

// Coalesced small-T planar Q8 matrix product: 1 wave per output row, 128-byte coalesced weight reads,
// exact FP32 scale & FMA (matches single-token decode numerics) with low VGPR count (~28-38 VGPRs).
template <int TFixed>
__global__ void rdna3_q8_mv_rows(const int8_t* __restrict__ q, const uint16_t* __restrict__ scales, int N, int K,
                                 const uint16_t* __restrict__ input, float* __restrict__ output, int ld) {
    constexpr int warps = 4;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*)(q + (size_t)row * K);
    const uint16_t* dr = scales + (size_t)row * (K >> 5);
    float acc[TFixed] = {};
    const int n_it = K >> 7;
#pragma unroll 1
    for (int it = 0; it < n_it; ++it) {
        const int off32 = it * 32 + lane;
        const uint32_t w = qr[off32];
        const float dv = (float)__builtin_bit_cast(_Float16, dr[it * 4 + (lane >> 3)]);
        const float w0 = dv * (float)(int8_t)(w & 0xFFu);
        const float w1 = dv * (float)(int8_t)((w >> 8) & 0xFFu);
        const float w2 = dv * (float)(int8_t)((w >> 16) & 0xFFu);
        const float w3 = dv * (float)(int8_t)(w >> 24);
        const size_t xoff = (size_t)off32 * 4;
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            const uint2 xh = *(const uint2*)(input + (size_t)t * K + xoff);
            const float2 x01 = h2u_to_f2(xh.x), x23 = h2u_to_f2(xh.y);
            acc[t] = fmaf(w0, x01.x, fmaf(w1, x01.y, fmaf(w2, x23.x, fmaf(w3, x23.y, acc[t]))));
        }
    }
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        const float s = wsum32(acc[t]);
        if (lane == 0) output[(size_t)t * ld + row] = s;
    }
}

// Fused separate-input sliding QKV/R/G planar Q8 projection: replaces 4 small launches with 1 full-occupancy launch.
template <int TFixed>
__global__ void rdna3_q8_mv_rows_sep(const int8_t* __restrict__ q, const uint16_t* __restrict__ scales, int N, int K,
                                     int HD, int KD, const uint16_t* __restrict__ xq, const uint16_t* __restrict__ xk,
                                     const uint16_t* __restrict__ xv, float* __restrict__ output, int ld) {
    constexpr int warps = 4;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* input = (row < HD || row >= HD + 2 * KD) ? xq : (row < HD + KD ? xk : xv);
    const uint32_t* qr = (const uint32_t*)(q + (size_t)row * K);
    const uint16_t* dr = scales + (size_t)row * (K >> 5);
    float acc[TFixed] = {};
    const int n_it = K >> 7;
#pragma unroll 1
    for (int it = 0; it < n_it; ++it) {
        const int off32 = it * 32 + lane;
        const uint32_t w = qr[off32];
        const float dv = (float)__builtin_bit_cast(_Float16, dr[it * 4 + (lane >> 3)]);
        const float w0 = dv * (float)(int8_t)(w & 0xFFu);
        const float w1 = dv * (float)(int8_t)((w >> 8) & 0xFFu);
        const float w2 = dv * (float)(int8_t)((w >> 16) & 0xFFu);
        const float w3 = dv * (float)(int8_t)(w >> 24);
        const size_t xoff = (size_t)off32 * 4;
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            const uint2 xh = *(const uint2*)(input + (size_t)t * K + xoff);
            const float2 x01 = h2u_to_f2(xh.x), x23 = h2u_to_f2(xh.y);
            acc[t] = fmaf(w0, x01.x, fmaf(w1, x01.y, fmaf(w2, x23.x, fmaf(w3, x23.y, acc[t]))));
        }
    }
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        const float s = wsum32(acc[t]);
        if (lane == 0) output[(size_t)t * ld + row] = s;
    }
}

// Coalesced small-T BF16 matrix product: Y [T][N] (ld) = X [T][K] BF16 . W [N][K]^T BF16, F32 sums.
template <int TFixed>
__global__ void rdna3_bf16_mv_rows(const uint16_t* __restrict__ W, int N, int K,
                                   const uint16_t* __restrict__ X, float* __restrict__ Y, int ld) {
    constexpr int warps = 4;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* wr = W + (size_t)row * K;
    float acc[TFixed] = {};
#pragma unroll 1
    for (int k = lane * 4; k < K; k += 128) {
        const uint2 w4 = *(const uint2*)(wr + k);
        const float w0 = __uint_as_float((w4.x & 0xFFFFu) << 16);
        const float w1 = __uint_as_float(w4.x & 0xFFFF0000u);
        const float w2 = __uint_as_float((w4.y & 0xFFFFu) << 16);
        const float w3 = __uint_as_float(w4.y & 0xFFFF0000u);
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            const uint2 x4 = *(const uint2*)(X + (size_t)t * K + k);
            acc[t] = fmaf(w0, __uint_as_float((x4.x & 0xFFFFu) << 16),
                     fmaf(w1, __uint_as_float(x4.x & 0xFFFF0000u),
                     fmaf(w2, __uint_as_float((x4.y & 0xFFFFu) << 16),
                     fmaf(w3, __uint_as_float(x4.y & 0xFFFF0000u), acc[t]))));
        }
    }
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        const float s = wsum32(acc[t]);
        if (lane == 0) Y[(size_t)t * ld + row] = s;
    }
}

// Single-token NVFP4 gate/up + SwiGLU: no LDS staging, 2 output channels per wave to reuse x in registers.
template <int Rows>
__global__ void rdna3_nvfp4_gu_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ tok,
                                      size_t eb0, const float* __restrict__ x, int C, int F,
                                      const float* __restrict__ sg, const float* __restrict__ su,
                                      float gcap, float ucap, uint16_t* __restrict__ h) {
    constexpr int warps = 4;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * warps + (threadIdx.x >> 5)) * Rows;
    if (first >= F) return;
    const size_t rb = (size_t)(C / 64) * 36;
    const uint8_t* base = blob[slot] + (size_t)first * rb;
    const float* xp = x + (size_t)tok[slot] * C;
    float ag[Rows] = {}, au[Rows] = {};
    const int nsub = C / 16;
    for (int sb = lane; sb < nsub; sb += 32) {
        const float4 x0 = *(const float4*)(xp + sb * 16 + 0);
        const float4 x1 = *(const float4*)(xp + sb * 16 + 4);
        const float4 x2 = *(const float4*)(xp + sb * 16 + 8);
        const float4 x3 = *(const float4*)(xp + sb * 16 + 12);
        const size_t blk_off = (size_t)(sb >> 2) * 36;
        const int sub = sb & 3;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (first + r >= F) break;
            const uint8_t* bg = base + (size_t)r * rb + blk_off;
            const float dg = nv_ue4m3_f32((uint8_t)(*(const uint32_t*)bg >> (8 * sub)));
            const uint32_t g0 = *(const uint32_t*)(bg + 4 + sub * 8);
            const uint32_t g1 = *(const uint32_t*)(bg + 8 + sub * 8);
            ag[r] += dg * (dot4_f4(nv4_f32(g0), x0) + dot4_f4(nv4_f32(g1), x1) +
                           dot4_f4(nv4_f32(g0 >> 4), x2) + dot4_f4(nv4_f32(g1 >> 4), x3));

            const uint8_t* bu = bg + eb0;
            const float du = nv_ue4m3_f32((uint8_t)(*(const uint32_t*)bu >> (8 * sub)));
            const uint32_t u0 = *(const uint32_t*)(bu + 4 + sub * 8);
            const uint32_t u1 = *(const uint32_t*)(bu + 8 + sub * 8);
            au[r] += du * (dot4_f4(nv4_f32(u0), x0) + dot4_f4(nv4_f32(u1), x1) +
                           dot4_f4(nv4_f32(u0 >> 4), x2) + dot4_f4(nv4_f32(u1 >> 4), x3));
        }
    }
    const float sgs = sg ? sg[slot] : 1.0f;
    const float sus = su ? su[slot] : 1.0f;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (first + r >= F) break;
        const float vg = wsum32(ag[r]);
        const float vu = wsum32(au[r]);
        if (lane == 0) {
            const float g = vg * sgs, u = vu * sus;
            const float v = gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap);
            h[(size_t)slot * F + first + r] = __builtin_bit_cast(uint16_t, (_Float16)v);
        }
    }
}

// Single-token NVFP4 down projection: FP16 activations via v_dot2_f32_f16, 4 rows per wave, zero LDS.
template <int Rows>
__global__ void rdna3_nvfp4_down_kernel(const uint8_t* const* __restrict__ blob, size_t off_down,
                                        const uint16_t* __restrict__ h, int C, int F, float* __restrict__ y) {
    constexpr int warps = 4;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * warps + (threadIdx.x >> 5)) * Rows;
    if (first >= C) return;
    const size_t rb = (size_t)(F / 64) * 36;
    const uint8_t* base = blob[slot] + off_down + (size_t)first * rb;
    const uint16_t* hp = h + (size_t)slot * F;
    float acc[Rows] = {};
    const int nsub = F / 16;
    for (int sb = lane; sb < nsub; sb += 32) {
        const uint4 x0 = *(const uint4*)(hp + sb * 16 + 0);
        const uint4 x1 = *(const uint4*)(hp + sb * 16 + 8);
        const size_t blk_off = (size_t)(sb >> 2) * 36;
        const int sub = sb & 3;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (first + r >= C) break;
            const uint8_t* b = base + (size_t)r * rb + blk_off;
            const _Float16 sc = nv_scale((uint8_t)(*(const uint32_t*)b >> (8 * sub)));
            const uint32_t q0 = *(const uint32_t*)(b + 4 + sub * 8);
            const uint32_t q1 = *(const uint32_t*)(b + 8 + sub * 8);
            const uint2 w01 = nv_four_halves(q0, sc);
            const uint2 w23 = nv_four_halves(q1, sc);
            const uint2 w45 = nv_four_halves(q0 >> 4, sc);
            const uint2 w67 = nv_four_halves(q1 >> 4, sc);
            float a = acc[r];
            a = fdot2_u32(w01.x, x0.x, a);
            a = fdot2_u32(w01.y, x0.y, a);
            a = fdot2_u32(w23.x, x0.z, a);
            a = fdot2_u32(w23.y, x0.w, a);
            a = fdot2_u32(w45.x, x1.x, a);
            a = fdot2_u32(w45.y, x1.y, a);
            a = fdot2_u32(w67.x, x1.z, a);
            a = fdot2_u32(w67.y, x1.w, a);
            acc[r] = a;
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (first + r >= C) break;
        const float v = wsum32(acc[r]);
        if (lane == 0) y[(size_t)slot * C + first + r] = v;
    }
}

// Grouped block-verify NVFP4 gate/up + SwiGLU:
// Separate __global__ per MaxT (2..8), FP16 activations via v_dot2_f32_f16, sequential gate/up decode,
// SGPR tok_g indexing -> ~36-44 VGPRs, 0 spills, high CU occupancy!
template <int MaxT>
__global__ void rdna3_block_gu_nvfp4_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n,
                                            const int* __restrict__ tok, const int* __restrict__ row, size_t off_up,
                                            int C, int F, const uint16_t* __restrict__ x16, const float* __restrict__ sg,
                                            const float* __restrict__ su, float gcap, float ucap,
                                            uint16_t* __restrict__ h) {
    constexpr int warps = 4;
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const int r = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= F) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* row_g = row + g * kGroupMax;
    const size_t rb = (size_t)(C / 64) * 36;
    const uint8_t* wg = bl + (size_t)r * rb;
    const uint8_t* wu = wg + off_up;
    float ag[MaxT] = {}, au[MaxT] = {};
    const int nsub = C / 16;
    for (int sb = lane; sb < nsub; sb += 32) {
        const size_t blk_off = (size_t)(sb >> 2) * 36;
        const int sub = sb & 3;
        const int koff = sb * 16;
        {
            const uint8_t* bg = wg + blk_off;
            const _Float16 dg = nv_scale((uint8_t)(*(const uint32_t*)bg >> (8 * sub)));
            const uint32_t g0 = *(const uint32_t*)(bg + 4 + sub * 8);
            const uint32_t g1 = *(const uint32_t*)(bg + 8 + sub * 8);
            const uint2 w01 = nv_four_halves(g0, dg);
            const uint2 w23 = nv_four_halves(g1, dg);
            const uint2 w45 = nv_four_halves(g0 >> 4, dg);
            const uint2 w67 = nv_four_halves(g1 >> 4, dg);
#pragma unroll
            for (int t = 0; t < MaxT; ++t) {
                if (t >= ng) break;
                const uint16_t* xp = x16 + (size_t)tok_g[t] * C + koff;
                const uint4 x0 = *(const uint4*)(xp + 0);
                const uint4 x1 = *(const uint4*)(xp + 8);
                float a = ag[t];
                a = fdot2_u32(w01.x, x0.x, a); a = fdot2_u32(w01.y, x0.y, a);
                a = fdot2_u32(w23.x, x0.z, a); a = fdot2_u32(w23.y, x0.w, a);
                a = fdot2_u32(w45.x, x1.x, a); a = fdot2_u32(w45.y, x1.y, a);
                a = fdot2_u32(w67.x, x1.z, a); a = fdot2_u32(w67.y, x1.w, a);
                ag[t] = a;
            }
        }
        {
            const uint8_t* bu = wu + blk_off;
            const _Float16 du = nv_scale((uint8_t)(*(const uint32_t*)bu >> (8 * sub)));
            const uint32_t u0 = *(const uint32_t*)(bu + 4 + sub * 8);
            const uint32_t u1 = *(const uint32_t*)(bu + 8 + sub * 8);
            const uint2 w01 = nv_four_halves(u0, du);
            const uint2 w23 = nv_four_halves(u1, du);
            const uint2 w45 = nv_four_halves(u0 >> 4, du);
            const uint2 w67 = nv_four_halves(u1 >> 4, du);
#pragma unroll
            for (int t = 0; t < MaxT; ++t) {
                if (t >= ng) break;
                const uint16_t* xp = x16 + (size_t)tok_g[t] * C + koff;
                const uint4 x0 = *(const uint4*)(xp + 0);
                const uint4 x1 = *(const uint4*)(xp + 8);
                float a = au[t];
                a = fdot2_u32(w01.x, x0.x, a); a = fdot2_u32(w01.y, x0.y, a);
                a = fdot2_u32(w23.x, x0.z, a); a = fdot2_u32(w23.y, x0.w, a);
                a = fdot2_u32(w45.x, x1.x, a); a = fdot2_u32(w45.y, x1.y, a);
                a = fdot2_u32(w67.x, x1.z, a); a = fdot2_u32(w67.y, x1.w, a);
                au[t] = a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MaxT; ++t) {
        if (t >= ng) break;
        const float vg = wsum32(ag[t]);
        const float vu = wsum32(au[t]);
        if (lane == 0) {
            const int slot = row_g[t];
            const float g_val = vg * (sg ? sg[slot] : 1.0f);
            const float u_val = vu * (su ? su[slot] : 1.0f);
            const float v = gcap * tanhf(g_val / gcap) * (1.f / (1.f + expf(-g_val))) * ucap * tanhf(u_val / ucap);
            h[(size_t)slot * F + r] = __builtin_bit_cast(uint16_t, (_Float16)v);
        }
    }
}

// Grouped block-verify NVFP4 down projection: separate __global__ per MaxT (2..8), ~32-40 VGPRs, 0 spills.
template <int MaxT>
__global__ void rdna3_block_down_nvfp4_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n,
                                              const int* __restrict__ row, size_t off_down, int C, int F,
                                              const uint16_t* __restrict__ h, float* __restrict__ y) {
    constexpr int warps = 4;
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const int r = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= C) return;
    const uint8_t* w = blob[g] + off_down;
    const int* row_g = row + g * kGroupMax;
    const size_t rb = (size_t)(F / 64) * 36;
    const uint8_t* wr = w + (size_t)r * rb;
    float acc[MaxT] = {};
    const int nsub = F / 16;
    for (int sb = lane; sb < nsub; sb += 32) {
        const uint8_t* b = wr + (size_t)(sb >> 2) * 36;
        const int sub = sb & 3;
        const int koff = sb * 16;
        const _Float16 sc = nv_scale((uint8_t)(*(const uint32_t*)b >> (8 * sub)));
        const uint32_t q0 = *(const uint32_t*)(b + 4 + sub * 8);
        const uint32_t q1 = *(const uint32_t*)(b + 8 + sub * 8);
        const uint2 w01 = nv_four_halves(q0, sc);
        const uint2 w23 = nv_four_halves(q1, sc);
        const uint2 w45 = nv_four_halves(q0 >> 4, sc);
        const uint2 w67 = nv_four_halves(q1 >> 4, sc);
#pragma unroll
        for (int t = 0; t < MaxT; ++t) {
            if (t >= ng) break;
            const uint16_t* hp = h + (size_t)row_g[t] * F + koff;
            const uint4 x0 = *(const uint4*)(hp + 0);
            const uint4 x1 = *(const uint4*)(hp + 8);
            float a = acc[t];
            a = fdot2_u32(w01.x, x0.x, a); a = fdot2_u32(w01.y, x0.y, a);
            a = fdot2_u32(w23.x, x0.z, a); a = fdot2_u32(w23.y, x0.w, a);
            a = fdot2_u32(w45.x, x1.x, a); a = fdot2_u32(w45.y, x1.y, a);
            a = fdot2_u32(w67.x, x1.z, a); a = fdot2_u32(w67.y, x1.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MaxT; ++t) {
        if (t >= ng) break;
        const float v = wsum32(acc[t]);
        if (lane == 0) y[(size_t)row_g[t] * C + r] = v;
    }
}

// Grouped block-verify 32-weight block (Q8_0 / IQ4_NL) gate/up + SwiGLU: scaled_byte_pair + v_dot2_f32_f16.
template <int Type, int MaxT>
__global__ void rdna3_block_gu_32_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n,
                                         const int* __restrict__ tok, const int* __restrict__ row, size_t off_up,
                                         int C, int F, const uint16_t* __restrict__ x16, const float* __restrict__ sg,
                                         const float* __restrict__ su, float gcap, float ucap,
                                         uint16_t* __restrict__ h) {
    static_assert(Type == kQ8_0 || Type == kIQ4_NL);
    constexpr int warps = 4;
    constexpr int kBlkBytes = Type == kIQ4_NL ? 18 : 34;
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const int r = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, word = lane & 3;
    if (r >= F) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* row_g = row + g * kGroupMax;
    const size_t rb = (size_t)(C / 32) * kBlkBytes;
    const uint8_t* wg = bl + (size_t)r * rb;
    const uint8_t* wu = wg + off_up;
    float ag[MaxT] = {}, au[MaxT] = {};
    for (int b = lane >> 2; b < C / 32; b += 8) {
        const int koff = b * 32 + word * 4;
        {
            const uint8_t* bg = wg + (size_t)b * kBlkBytes;
            uint16_t dg16; uint32_t gl, gh;
            __builtin_memcpy(&dg16, bg, 2);
            if constexpr (Type == kIQ4_NL) {
                uint32_t packed;
                __builtin_memcpy(&packed, bg + 2 + word * 4, 4);
                gl = iq4_values(packed);
                gh = iq4_values(packed >> 4);
            } else {
                __builtin_memcpy(&gl, bg + 2 + word * 4, 4);
                __builtin_memcpy(&gh, bg + 18 + word * 4, 4);
            }
            const _Float16 dg = __builtin_bit_cast(_Float16, dg16);
            const uint32_t w0 = scaled_byte_pair<false>(gl, dg), w1 = scaled_byte_pair<true>(gl, dg);
            const uint32_t w2 = scaled_byte_pair<false>(gh, dg), w3 = scaled_byte_pair<true>(gh, dg);
#pragma unroll
            for (int t = 0; t < MaxT; ++t) {
                if (t >= ng) break;
                const uint16_t* xp = x16 + (size_t)tok_g[t] * C + koff;
                const uint2 xl = *(const uint2*)(xp);
                const uint2 xh = *(const uint2*)(xp + 16);
                float a = ag[t];
                a = fdot2_u32(w0, xl.x, a); a = fdot2_u32(w1, xl.y, a);
                a = fdot2_u32(w2, xh.x, a); a = fdot2_u32(w3, xh.y, a);
                ag[t] = a;
            }
        }
        {
            const uint8_t* bu = wu + (size_t)b * kBlkBytes;
            uint16_t du16; uint32_t ul, uh;
            __builtin_memcpy(&du16, bu, 2);
            if constexpr (Type == kIQ4_NL) {
                uint32_t packed;
                __builtin_memcpy(&packed, bu + 2 + word * 4, 4);
                ul = iq4_values(packed);
                uh = iq4_values(packed >> 4);
            } else {
                __builtin_memcpy(&ul, bu + 2 + word * 4, 4);
                __builtin_memcpy(&uh, bu + 18 + word * 4, 4);
            }
            const _Float16 du = __builtin_bit_cast(_Float16, du16);
            const uint32_t w0 = scaled_byte_pair<false>(ul, du), w1 = scaled_byte_pair<true>(ul, du);
            const uint32_t w2 = scaled_byte_pair<false>(uh, du), w3 = scaled_byte_pair<true>(uh, du);
#pragma unroll
            for (int t = 0; t < MaxT; ++t) {
                if (t >= ng) break;
                const uint16_t* xp = x16 + (size_t)tok_g[t] * C + koff;
                const uint2 xl = *(const uint2*)(xp);
                const uint2 xh = *(const uint2*)(xp + 16);
                float a = au[t];
                a = fdot2_u32(w0, xl.x, a); a = fdot2_u32(w1, xl.y, a);
                a = fdot2_u32(w2, xh.x, a); a = fdot2_u32(w3, xh.y, a);
                au[t] = a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MaxT; ++t) {
        if (t >= ng) break;
        const float vg = wsum32(ag[t]);
        const float vu = wsum32(au[t]);
        if (lane == 0) {
            const int slot = row_g[t];
            const float g_val = vg * (sg ? sg[slot] : 1.0f);
            const float u_val = vu * (su ? su[slot] : 1.0f);
            const float v = gcap * tanhf(g_val / gcap) * (1.f / (1.f + expf(-g_val))) * ucap * tanhf(u_val / ucap);
            h[(size_t)slot * F + r] = __builtin_bit_cast(uint16_t, (_Float16)v);
        }
    }
}

// Grouped block-verify 32-weight block (Q8_0 / IQ4_NL) down projection: scaled_byte_pair + v_dot2_f32_f16.
template <int Type, int MaxT>
__global__ void rdna3_block_down_32_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n,
                                           const int* __restrict__ row, size_t off_down, int C, int F,
                                           const uint16_t* __restrict__ h, float* __restrict__ y) {
    static_assert(Type == kQ8_0 || Type == kIQ4_NL);
    constexpr int warps = 4;
    constexpr int kBlkBytes = Type == kIQ4_NL ? 18 : 34;
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const int r = blockIdx.x * warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, word = lane & 3;
    if (r >= C) return;
    const uint8_t* w = blob[g] + off_down;
    const int* row_g = row + g * kGroupMax;
    const size_t rb = (size_t)(F / 32) * kBlkBytes;
    const uint8_t* wr = w + (size_t)r * rb;
    float acc[MaxT] = {};
    for (int b = lane >> 2; b < F / 32; b += 8) {
        const uint8_t* blk = wr + (size_t)b * kBlkBytes;
        uint16_t d16; uint32_t wl, wh;
        __builtin_memcpy(&d16, blk, 2);
        if constexpr (Type == kIQ4_NL) {
            uint32_t packed;
            __builtin_memcpy(&packed, blk + 2 + word * 4, 4);
            wl = iq4_values(packed);
            wh = iq4_values(packed >> 4);
        } else {
            __builtin_memcpy(&wl, blk + 2 + word * 4, 4);
            __builtin_memcpy(&wh, blk + 18 + word * 4, 4);
        }
        const _Float16 d = __builtin_bit_cast(_Float16, d16);
        const uint32_t w0 = scaled_byte_pair<false>(wl, d), w1 = scaled_byte_pair<true>(wl, d);
        const uint32_t w2 = scaled_byte_pair<false>(wh, d), w3 = scaled_byte_pair<true>(wh, d);
        const int koff = b * 32 + word * 4;
#pragma unroll
        for (int t = 0; t < MaxT; ++t) {
            if (t >= ng) break;
            const uint16_t* hp = h + (size_t)row_g[t] * F + koff;
            const uint2 hl = *(const uint2*)(hp);
            const uint2 hh = *(const uint2*)(hp + 16);
            float a = acc[t];
            a = fdot2_u32(w0, hl.x, a); a = fdot2_u32(w1, hl.y, a);
            a = fdot2_u32(w2, hh.x, a); a = fdot2_u32(w3, hh.y, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MaxT; ++t) {
        if (t >= ng) break;
        const float v = wsum32(acc[t]);
        if (lane == 0) y[(size_t)row_g[t] * C + r] = v;
    }
}

}  // namespace

void rdna3_dense_gemm(const uint16_t* weights, int N, int K, const uint16_t* input, int T, float* output,
                      int ld, cudaStream_t stream) {
    launch_tiled<kDenseF16>(dim3((N + 63) / 64, (T + 63) / 64), weights, nullptr, 0, nullptr,
                            input, N, K, T, output, ld, stream);
    check_launch("dense_half_gemm");
}

void rdna3_planar_gemm(const int8_t* weights, const uint16_t* scales, int N, int K, const uint16_t* input, int T,
                       float* output, int ld, cudaStream_t stream) {
    launch_tiled<kPlanarQ8>(dim3((N + 63) / 64, (T + 63) / 64), nullptr, nullptr, 0, nullptr,
                            input, N, K, T, output, ld, stream, weights, scales);
    check_launch("dense_planar_gemm");
}

void q8_tile(const int8_t* q, const uint16_t* scales, int N, int K, const uint16_t* input, int T,
             float* output, int ld, cudaStream_t stream) {
    static const bool use_mv = !getenv("IK2_Q8_TILE_MV") || atoi(getenv("IK2_Q8_TILE_MV"));
    if (use_mv && T >= 1 && T <= 8 && K % 128 == 0 && ((uintptr_t)q & 3) == 0 && ((uintptr_t)input & 7) == 0) {
        const unsigned blocks = (unsigned)((N + 3) / 4);
        switch (T) {
            case 1: rdna3_q8_mv_rows<1><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 2: rdna3_q8_mv_rows<2><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 3: rdna3_q8_mv_rows<3><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 4: rdna3_q8_mv_rows<4><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 5: rdna3_q8_mv_rows<5><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 6: rdna3_q8_mv_rows<6><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 7: rdna3_q8_mv_rows<7><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
            case 8: rdna3_q8_mv_rows<8><<<blocks, 128, 0, stream>>>(q, scales, N, K, input, output, ld); break;
        }
        check_launch("rdna3_q8_mv_rows");
        return;
    }
    dense_q8_wmma<<<dim3((N + 63) / 64, (T + 15) / 16), 128, 0, stream>>>(q, scales, N, K, input, T, output, ld);
    check_launch("dense_q8_wmma");
}

void rdna3_bf16_gemm(const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy,
                     cudaStream_t stream) {
    const unsigned blocks = (unsigned)((N + 3) / 4);
    switch (T) {
        case 1: rdna3_bf16_mv_rows<1><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 2: rdna3_bf16_mv_rows<2><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 3: rdna3_bf16_mv_rows<3><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 4: rdna3_bf16_mv_rows<4><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 5: rdna3_bf16_mv_rows<5><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 6: rdna3_bf16_mv_rows<6><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 7: rdna3_bf16_mv_rows<7><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
        case 8: rdna3_bf16_mv_rows<8><<<blocks, 128, 0, stream>>>(W, N, K, X, Y, ldy); break;
    }
    check_launch("rdna3_bf16_gemm");
}

void rdna3_q8_tile_qkv_sep(const int8_t* q, const uint16_t* scales, int N, int K, int HD, int KD, const uint16_t* xq,
                           const uint16_t* xk, const uint16_t* xv, int T, float* y, int ldy, cudaStream_t stream) {
    const unsigned blocks = (unsigned)((N + 3) / 4);
    switch (T) {
        case 1: rdna3_q8_mv_rows_sep<1><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 2: rdna3_q8_mv_rows_sep<2><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 3: rdna3_q8_mv_rows_sep<3><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 4: rdna3_q8_mv_rows_sep<4><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 5: rdna3_q8_mv_rows_sep<5><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 6: rdna3_q8_mv_rows_sep<6><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 7: rdna3_q8_mv_rows_sep<7><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
        case 8: rdna3_q8_mv_rows_sep<8><<<blocks, 128, 0, stream>>>(q, scales, N, K, HD, KD, xq, xk, xv, y, ldy); break;
    }
    check_launch("rdna3_q8_tile_qkv_sep");
}

void rdna3_nvfp4_token_gu(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s) {
    if (S <= 0) return;
    constexpr int rows = 2, warps = 4;
    rdna3_nvfp4_gu_kernel<rows><<<dim3((unsigned)((F + rows * warps - 1) / (rows * warps)), (unsigned)S),
                                  warps * 32, 0, s>>>(blob, tok, eb0, x, C, F, sg, su, gcap, ucap, h);
    check_launch("rdna3_nvfp4_token_gu");
}

void rdna3_nvfp4_token_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F,
                            float* y, cudaStream_t s) {
    if (S <= 0) return;
    constexpr int rows = 4, warps = 4;
    rdna3_nvfp4_down_kernel<rows><<<dim3((unsigned)((C + rows * warps - 1) / (rows * warps)), (unsigned)S),
                                    warps * 32, 0, s>>>(blob, off_down, h, C, F, y);
    check_launch("rdna3_nvfp4_token_down");
}

void rdna3_experts_block_gu(int type, const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                            int T, size_t off_up, int C, int F, const uint16_t* x16, const float* sg, const float* su,
                            float gcap, float ucap, uint16_t* h, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned)((F + 3) / 4), (unsigned)G);
    const int mt = T < 1 ? 1 : (T > kGroupMax ? kGroupMax : T);
    if (type == kNVFP4) {
        switch (mt) {
            case 1: rdna3_block_gu_nvfp4_kernel<1><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 2: rdna3_block_gu_nvfp4_kernel<2><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 3: rdna3_block_gu_nvfp4_kernel<3><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 4: rdna3_block_gu_nvfp4_kernel<4><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 5: rdna3_block_gu_nvfp4_kernel<5><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 6: rdna3_block_gu_nvfp4_kernel<6><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 7: rdna3_block_gu_nvfp4_kernel<7><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            default: rdna3_block_gu_nvfp4_kernel<8><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
        }
    } else if (type == kQ8_0) {
        switch (mt) {
            case 1: rdna3_block_gu_32_kernel<kQ8_0, 1><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 2: rdna3_block_gu_32_kernel<kQ8_0, 2><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 3: rdna3_block_gu_32_kernel<kQ8_0, 3><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 4: rdna3_block_gu_32_kernel<kQ8_0, 4><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 5: rdna3_block_gu_32_kernel<kQ8_0, 5><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 6: rdna3_block_gu_32_kernel<kQ8_0, 6><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 7: rdna3_block_gu_32_kernel<kQ8_0, 7><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            default: rdna3_block_gu_32_kernel<kQ8_0, 8><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
        }
    } else if (type == kIQ4_NL) {
        switch (mt) {
            case 1: rdna3_block_gu_32_kernel<kIQ4_NL, 1><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 2: rdna3_block_gu_32_kernel<kIQ4_NL, 2><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 3: rdna3_block_gu_32_kernel<kIQ4_NL, 3><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 4: rdna3_block_gu_32_kernel<kIQ4_NL, 4><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 5: rdna3_block_gu_32_kernel<kIQ4_NL, 5><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 6: rdna3_block_gu_32_kernel<kIQ4_NL, 6><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            case 7: rdna3_block_gu_32_kernel<kIQ4_NL, 7><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
            default: rdna3_block_gu_32_kernel<kIQ4_NL, 8><<<g, 128, 0, s>>>(blob, n, tok, row, off_up, C, F, x16, sg, su, gcap, ucap, h); break;
        }
    }
    check_launch("rdna3_experts_block_gu");
}

void rdna3_experts_block_down(int type, const uint8_t* const* blob, const int* n, const int* row, int G, int T,
                              size_t off_down, int C, int F, const uint16_t* h, float* y, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned)((C + 3) / 4), (unsigned)G);
    const int mt = T < 1 ? 1 : (T > kGroupMax ? kGroupMax : T);
    if (type == kNVFP4) {
        switch (mt) {
            case 1: rdna3_block_down_nvfp4_kernel<1><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 2: rdna3_block_down_nvfp4_kernel<2><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 3: rdna3_block_down_nvfp4_kernel<3><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 4: rdna3_block_down_nvfp4_kernel<4><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 5: rdna3_block_down_nvfp4_kernel<5><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 6: rdna3_block_down_nvfp4_kernel<6><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 7: rdna3_block_down_nvfp4_kernel<7><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            default: rdna3_block_down_nvfp4_kernel<8><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
        }
    } else if (type == kQ8_0) {
        switch (mt) {
            case 1: rdna3_block_down_32_kernel<kQ8_0, 1><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 2: rdna3_block_down_32_kernel<kQ8_0, 2><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 3: rdna3_block_down_32_kernel<kQ8_0, 3><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 4: rdna3_block_down_32_kernel<kQ8_0, 4><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 5: rdna3_block_down_32_kernel<kQ8_0, 5><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 6: rdna3_block_down_32_kernel<kQ8_0, 6><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 7: rdna3_block_down_32_kernel<kQ8_0, 7><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            default: rdna3_block_down_32_kernel<kQ8_0, 8><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
        }
    } else if (type == kIQ4_NL) {
        switch (mt) {
            case 1: rdna3_block_down_32_kernel<kIQ4_NL, 1><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 2: rdna3_block_down_32_kernel<kIQ4_NL, 2><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 3: rdna3_block_down_32_kernel<kIQ4_NL, 3><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 4: rdna3_block_down_32_kernel<kIQ4_NL, 4><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 5: rdna3_block_down_32_kernel<kIQ4_NL, 5><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 6: rdna3_block_down_32_kernel<kIQ4_NL, 6><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            case 7: rdna3_block_down_32_kernel<kIQ4_NL, 7><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
            default: rdna3_block_down_32_kernel<kIQ4_NL, 8><<<g, 128, 0, s>>>(blob, n, row, off_down, C, F, h, y); break;
        }
    }
    check_launch("rdna3_experts_block_down");
}

void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int slots, size_t offset, const uint16_t* x,
                                int C, int F, const float* sg, const float* su, float gate_cap, float up_cap,
                                uint16_t* hidden, cudaStream_t stream) {
    if (slots <= 0) return;
    nv_token_wmma<true><<<dim3(F / 16, slots), 32 * nv_token_waves<true>, 0, stream>>>(blob, offset, x, C, F, sg, su, gate_cap, up_cap, hidden, nullptr);
    check_launch("nv_token_wmma_gate_up");
}

void experts_down_mma(const uint8_t* const* blob, int slots, size_t offset, const uint16_t* hidden,
                      int C, int F, float* output, cudaStream_t stream) {
    if (slots <= 0) return;
    nv_token_wmma<false><<<dim3(C / 16, slots), 32 * nv_token_waves<false>, 0, stream>>>(blob, offset, hidden, F, C, nullptr, nullptr, 0.f, 0.f, nullptr, output);
    check_launch("nv_token_wmma_down");
}

void rdna3_expert_groups(const int* top, const float* weights, int slots, int K, int experts, const float* gs,
                         const float* us, const float* ds, ExpertTiles tiles, int* tokens, int* inv, float* gate,
                         float* up, float* weight, cudaStream_t stream) {
    group_expert_rows<<<1, 256, 3 * experts * sizeof(int), stream>>>(top, weights, slots, K, experts, gs, us, ds,
        tiles, tokens, inv, gate, up, weight);
    check_launch("group_expert_rows");
}

template <int Type>
void launch_expert_gemm(const uint8_t* const* experts, size_t offset, ExpertTiles tiles, const uint16_t* input,
                        int K, int N, float* output, cudaStream_t stream) {
    const dim3 grid((N + 63) / 64, tiles.capacity);
    static const bool use_lds = getenv("IK2_EXPERT_TILE_LDS") && atoi(getenv("IK2_EXPERT_TILE_LDS"));
    if (tiles.rows == 64 && use_lds)
        launch_tiled<Type>(grid, nullptr, experts, offset, tiles.data, input, N, K, 0, output, N, stream);
    else if (tiles.rows == 16) expert_wmma_gemm<Type, 16><<<grid, 128, 0, stream>>>(experts, offset, tiles.data, input, K, N, output);
    else if (tiles.rows == 32) expert_wmma_gemm<Type, 32><<<grid, 128, 0, stream>>>(experts, offset, tiles.data, input, K, N, output);
    else if (tiles.rows == 64) expert_wmma_gemm<Type, 64><<<grid, 128, 0, stream>>>(experts, offset, tiles.data, input, K, N, output);
    else {
        std::fprintf(stderr, "Unsupported expert tile height %d\n", tiles.rows);
        std::abort();
    }
    check_launch("expert_wmma_gemm");
}

void rdna3_expert_gemm(int type, const uint8_t* const* experts, size_t offset, ExpertTiles tiles,
                       const uint16_t* input, int K, int N, float* output, cudaStream_t stream) {
    if (type == kIQ4_NL) launch_expert_gemm<kIQ4_NL>(experts, offset, tiles, input, K, N, output, stream);
    else if (type == kNVFP4) launch_expert_gemm<kNVFP4>(experts, offset, tiles, input, K, N, output, stream);
    else if (type == kQ8_0) launch_expert_gemm<kQ8_0>(experts, offset, tiles, input, K, N, output, stream);
    else {
        std::fprintf(stderr, "rdna3_expert_gemm: unsupported expert type %d\n", type);
        std::abort();
    }
}
void rdna3_expert_swiglu(int type, const uint8_t* const* experts, ExpertTiles tiles, const uint16_t* input,
                         int K, int F, const float* sg, const float* su, float gate_cap, float up_cap,
                         uint16_t* hidden, cudaStream_t stream) {
    const dim3 grid((F + 31) / 32, tiles.capacity);
    const ExpertActivation act{sg, su, gate_cap, up_cap, hidden};
    if (type == kIQ4_NL)
        launch_tiled<kIQ4_NL, true>(grid, nullptr, experts, 0, tiles.data, input, F, K, 0, nullptr, 0, stream, nullptr, nullptr, act);
    else if (type == kNVFP4)
        launch_tiled<kNVFP4, true>(grid, nullptr, experts, 0, tiles.data, input, F, K, 0, nullptr, 0, stream, nullptr, nullptr, act);
    else if (type == kQ8_0)
        launch_tiled<kQ8_0, true>(grid, nullptr, experts, 0, tiles.data, input, F, K, 0, nullptr, 0, stream, nullptr, nullptr, act);
    else {
        std::fprintf(stderr, "rdna3_expert_swiglu: unsupported expert type %d\n", type);
        std::abort();
    }
    check_launch("rdna3_expert_swiglu");
}
}  // namespace ik2::k
