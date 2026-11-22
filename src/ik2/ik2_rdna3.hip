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

__device__ uint32_t nv_values(uint32_t codes) {
    const uint32_t p = codes & 0x07070707u;
    const uint32_t positive = __builtin_amdgcn_perm(0x0c080604u, 0x03020100u, p);
    const uint32_t negative = __builtin_amdgcn_perm(0xf4f8fafcu, 0xfdfeff00u, p);
    return __builtin_amdgcn_perm(negative, positive, ((codes >> 1) & 0x04040404u) | 0x03020100u);
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
            packed = nv_values(packed >> ((i / 2) * 4));
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

// Sixteen output channels share one input. Two WMMA rows carry the high and low
// FP16 limbs of that input. Four waves accumulate disjoint 64-weight blocks.
template <bool Gate>
__global__ void nv_token_wmma(const uint8_t* const* experts, size_t offset, const uint16_t* input,
                              int K, int N, const float* sg, const float* su, float gate_cap,
                              float up_cap, uint16_t* hidden, float* output) {
    constexpr int splits = 4;
    __shared__ float partial[splits][Gate ? 2 : 1][16];
    const int lane = threadIdx.x & 31, col = lane & 15, split = threadIdx.x / 32;
    const int slot = blockIdx.y, row = blockIdx.x * 16 + col;
    const uint8_t* w = experts[slot] + expert_byte_offset<kNVFP4>((size_t)row * K) + (Gate ? 0 : offset);
    const uint16_t* x = input + (Gate ? (lane & 1) * K : (size_t)slot * K);
    Float8 first = {}, second = {};
    for (int block = split * 64; block < K; block += splits * 64) {
#pragma unroll
        for (int j = 0; j < 64; j += 16) {
            Half16 a;
            __builtin_memcpy(&a, x + block + j, sizeof(a));
            first = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, expert_fragment<kNVFP4>(w, block + j), first);
            if constexpr (Gate)
                second = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, expert_fragment<kNVFP4>(w + offset, block + j), second);
        }
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
    dense_q8_wmma<<<dim3((N + 63) / 64, (T + 15) / 16), 128, 0, stream>>>(q, scales, N, K, input, T, output, ld);
    check_launch("dense_q8_wmma");
}

void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int slots, size_t offset, const uint16_t* x,
                                int C, int F, const float* sg, const float* su, float gate_cap, float up_cap,
                                uint16_t* hidden, cudaStream_t stream) {
    if (slots <= 0) return;
    nv_token_wmma<true><<<dim3(F / 16, slots), 128, 0, stream>>>(blob, offset, x, C, F, sg, su, gate_cap, up_cap, hidden, nullptr);
    check_launch("nv_token_wmma_gate_up");
}

void experts_down_mma(const uint8_t* const* blob, int slots, size_t offset, const uint16_t* hidden,
                      int C, int F, float* output, cudaStream_t stream) {
    if (slots <= 0) return;
    nv_token_wmma<false><<<dim3(C / 16, slots), 128, 0, stream>>>(blob, offset, hidden, F, C, nullptr, nullptr, 0.f, 0.f, nullptr, output);
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
