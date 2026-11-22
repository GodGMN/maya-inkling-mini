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

// Each lane owns a consecutive group of sixteen weights in an output row.
template <int Type>
__device__ Half16 expert_fragment(const uint8_t* block, int half) {
    HalfWords out;
    const _Float16 scale = __builtin_bit_cast(_Float16, *(const uint16_t*)block);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t packed;
        if constexpr (Type == kIQ4_NL) {
            __builtin_memcpy(&packed, block + 2 + i * 4, 4);
            packed = iq4_values(packed >> (half * 4));
        } else {
            __builtin_memcpy(&packed, block + 2 + half * 16 + i * 4, 4);
        }
        out[i * 2] = scaled_byte_pair<false>(packed, scale);
        out[i * 2 + 1] = scaled_byte_pair<true>(packed, scale);
    }
    return __builtin_bit_cast(Half16, out);
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
    constexpr int block_bytes = Type == kIQ4_NL ? 18 : 34;
    const auto [expert, first, end] = tiles[blockIdx.y];
    if (first >= end) return;
    const int lane = threadIdx.x & 31, lane16 = lane & 15;
    const int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + lane16;
    if (col - lane16 >= N) return;
    const int token = first + lane16;
    const uint8_t* w = experts[expert] + offset + (size_t)col * (K / 32) * block_bytes;
    Float8 result[Rows / 16] = {};
    for (int k = 0; k < K; k += 32) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const Half16 b = expert_fragment<Type>(w + (k / 32) * block_bytes, half);
#pragma unroll
            for (int m = 0; m < Rows / 16; ++m) {
                if (m && first + m * 16 >= end) continue;
                Half16 a = {};
                if (token + m * 16 < end)
                    __builtin_memcpy(&a, input + (size_t)(token + m * 16) * K + k + half * 16, sizeof(a));
                result[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, result[m]);
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
template <int Type, int Depth>
__global__ void tiled_gemm(const uint16_t* dense, const uint8_t* const* experts, size_t offset, const ExpertTile* tiles,
                           const uint16_t* input, int N, int K, int T, float* output, int ld,
                           const int8_t* planar = nullptr, const uint16_t* scales = nullptr) {
    constexpr int vectors = Depth / 8, pitch = vectors + 1;
    __shared__ uint4 a[64][pitch], b[64][pitch];
    const int tid = threadIdx.x, lane = tid & 31, l = lane & 15, wave = tid / 32;
    int row0 = blockIdx.y * 64;
    const uint8_t* quant = nullptr;
    if constexpr (Type == kIQ4_NL || Type == kQ8_0) {
        const auto tile = tiles[blockIdx.y];
        if (tile.first >= tile.end) return;
        row0 = tile.first;
        T = tile.end;
        quant = experts[tile.expert] + offset;
    }
    const int col0 = blockIdx.x * 64;
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
            constexpr int bytes = Type == kIQ4_NL ? 18 : 34;
#pragma unroll
            for (int f = tid; f < 64 * (Depth / 16); f += 128) {
                const int row = f / (Depth / 16), frag = f % (Depth / 16);
                Half16 bv = {};
                if (col0 + row < N) {
                    const size_t base = (size_t)(col0 + row) * K + k0 + frag * 16;
                    if constexpr (Type == kPlanarQ8) {
                        const _Float16 scale = __builtin_bit_cast(_Float16, scales[base / 32]);
                        bv = planar_fragment(planar + base, scale);
                    } else {
                        bv = expert_fragment<Type>(quant + (base / 32) * bytes, frag % 2);
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
                    accum[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ax[m], bx[n], accum[m][n]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int col = col0 + n0 + n * 16 + l;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + m0 + m * 16 + 2 * i + (lane >> 4);
                if (row < T && col < N) output[(size_t)row * ld + col] = accum[m][n][i];
            }
        }
}

template <int Type>
void launch_tiled(dim3 grid, const uint16_t* dense, const uint8_t* const* experts, size_t offset,
                  const ExpertTile* tiles, const uint16_t* x, int N, int K, int T, float* y, int ld,
                  cudaStream_t stream, const int8_t* planar = nullptr, const uint16_t* scales = nullptr) {
    static const bool wide = getenv("IK2_WMMA_K") && atoi(getenv("IK2_WMMA_K")) == 64;
    if (wide && K % 64 == 0)
        tiled_gemm<Type, 64><<<grid, 128, 0, stream>>>(dense, experts, offset, tiles, x, N, K, T, y, ld, planar, scales);
    else
        tiled_gemm<Type, 32><<<grid, 128, 0, stream>>>(dense, experts, offset, tiles, x, N, K, T, y, ld, planar, scales);
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
    else if (type == kQ8_0) launch_expert_gemm<kQ8_0>(experts, offset, tiles, input, K, N, output, stream);
    else {
        std::fprintf(stderr, "rdna3_expert_gemm: unsupported expert type %d\n", type);
        std::abort();
    }
}
}  // namespace ik2::k
