// Inkling-2-Mini DFlash2. All GEMM outputs enter the next operator rounded to BF16.
#pragma once
#include "ik2_draft.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace ik2::draft::df2 {
constexpr int width = 2304, groups = width / 16, dyn_width = 4 * groups;
constexpr int vocab = 200058, codebook_rows = 201024, rank = 256, top = kCandidates, chunk = 4096;
constexpr int parts = (vocab + chunk - 1) / chunk;

#ifndef STRATA_USE_HIP
// Store one m16k16 weight fragment in the order consumed by the 32 MMA lanes.
static __global__ void pack_projection(const uint16_t* w, int k, uint4* packed) {
    const int lane = threadIdx.x % 32, r = blockIdx.x * 16 + lane / 4;
    const int tile = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    if (tile >= k / 16) return;
    const uint16_t* p = w + size_t(r) * k + 16 * tile + 2 * (lane % 4);
    const auto word = [](const uint16_t* v) { return *reinterpret_cast<const unsigned*>(v); };
    packed[(size_t(blockIdx.x) * (k / 16) + tile) * 32 + lane] =
        make_uint4(word(p), word(p + 8*k), word(p + 8), word(p + 8*k + 8));
}

// Fragment bytes store biased integers. Converting two lanes at once via 1024 + u
// is exact in FP16. Each output channel has its own FP32 scale.
template<int Bits>
__device__ inline unsigned unpack_pair(unsigned packed) {
    constexpr unsigned mask = (1u << Bits) - 1;
    unsigned lanes = 0x64006400u | (packed & mask) | (((packed >> Bits) & mask) << 16);
    unsigned result;
    constexpr unsigned bias = 0x6400u + (1u << (Bits - 1));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(result) : "r"(lanes), "r"(bias | (bias << 16)));
    return result;
}

// Each output uses four K partitions and the same ordered MMA reductions for every batch size.
// A warp keeps its weight fragment in registers across all of the tile's 8-row input panels.
template <int Panels, bool Half = false, int Bits = 16>
static __global__ void project(const uint16_t* weight, const uint16_t* x, int n, int k, int rows,
                               float* y, int ld, const float* scales) {
    const int lane = threadIdx.x % 32, split = threadIdx.x / 32;
    const int g = lane / 4, pair = 2 * (lane % 4), col = blockIdx.x * 16 + g;
    const int first = blockIdx.y * Panels * 8;
    float acc[Panels][4]{};
    for (int i = 16 * split; i < k; i += 64) {
        uint4 a;
        if constexpr (Bits < 16) {
            const size_t index = (size_t(blockIdx.x) * k / 16 + i / 16) * 32 + lane;
            if constexpr (Bits == 8) {
                uint2 b = reinterpret_cast<const uint2*>(weight)[index];
                a = make_uint4(unpack_pair<8>(b.x), unpack_pair<8>(b.x >> 16),
                               unpack_pair<8>(b.y), unpack_pair<8>(b.y >> 16));
            } else {
                unsigned b = reinterpret_cast<const unsigned*>(weight)[index];
                a = make_uint4(unpack_pair<4>(b), unpack_pair<4>(b >> 8),
                               unpack_pair<4>(b >> 16), unpack_pair<4>(b >> 24));
            }
        } else if constexpr (Half) {
            const uint16_t* w0 = weight + size_t(min(col, n - 1)) * k + i + pair;
            const uint16_t* w1 = weight + size_t(min(col + 8, n - 1)) * k + i + pair;
            a = make_uint4(*reinterpret_cast<const unsigned*>(w0), *reinterpret_cast<const unsigned*>(w1),
                           *reinterpret_cast<const unsigned*>(w0 + 8), *reinterpret_cast<const unsigned*>(w1 + 8));
        } else {
            a = reinterpret_cast<const uint4*>(weight)[(size_t(blockIdx.x) * k / 16 + i / 16) * 32 + lane];
        }
#pragma unroll
        for (int p = 0; p < Panels; ++p) {
            const uint16_t* v = x + size_t(min(first + 8*p + g, rows-1)) * k + i + pair;
            const unsigned b0 = *reinterpret_cast<const unsigned*>(v);
            const unsigned b1 = *reinterpret_cast<const unsigned*>(v + 8);
#define IK2_DRAFT_MMA(dtype) asm volatile( \
    "mma.sync.aligned.m16n8k16.row.col.f32." dtype "." dtype ".f32 " \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
    : "+f"(acc[p][0]), "+f"(acc[p][1]), "+f"(acc[p][2]), "+f"(acc[p][3]) \
    : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1))
            if constexpr (Half || Bits < 16) { IK2_DRAFT_MMA("f16"); }
            else { IK2_DRAFT_MMA("bf16"); }
#undef IK2_DRAFT_MMA
        }
    }
    __shared__ float4 partial[4][32];
#pragma unroll
    for (int p = 0; p < Panels; ++p) {
        partial[split][lane] = make_float4(acc[p][0], acc[p][1], acc[p][2], acc[p][3]);
        __syncthreads();
        if (split == 0) {
            float4 v = partial[0][lane];
#pragma unroll
            for (int j = 1; j < 4; ++j) {
                const float4 a = partial[j][lane];
                v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
            }
            if constexpr (Bits < 16) {
                v.x *= scales[col]; v.y *= scales[col];
                v.z *= scales[col+8]; v.w *= scales[col+8];
            }
            const int t = first + 8*p + pair;
            if (t < rows && col < n) y[size_t(t) * ld + col] = v.x;
            if (t+1 < rows && col < n) y[size_t(t+1) * ld + col] = v.y;
            if (t < rows && col+8 < n) y[size_t(t) * ld + col+8] = v.z;
            if (t+1 < rows && col+8 < n) y[size_t(t+1) * ld + col+8] = v.w;
        }
        __syncthreads();
    }
}

template <bool Half = false, int Bits = 16>
static void project_rows(const uint16_t* weight, const uint16_t* x, int n, int k, int rows, float* y,
                         int ld, cudaStream_t stream, const float* scales = nullptr) {
    const int blocks = (n + 15) / 16;
    if (rows <= 8) project<1, Half, Bits><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld, scales);
    else if (rows <= 16) project<2, Half, Bits><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld, scales);
    else if (rows <= 32) project<4, Half, Bits><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld, scales);
    else project<8, Half, Bits><<<dim3(blocks, (rows+63)/64), 128, 0, stream>>>(weight, x, n, k, rows, y, ld, scales);
}
#endif

__device__ inline uint16_t bits(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
__device__ inline float expand(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ inline float rounded(float x) { return expand(bits(x)); }

// Quantize on the loading stream. All backbone shapes are multiples of 16 by 64.
template<int Bits>
static __global__ void quantize_rows(const uint16_t* input, int k, uint8_t* packed, float* scales) {
    const int row = blockIdx.x, lane = threadIdx.x;
    input += size_t(row) * k;
    __shared__ float peak[256];
    float m = 0;
    for (int j = lane; j < k; j += 256) m = fmaxf(m, fabsf(expand(input[j])));
    peak[lane] = m;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (lane < n) peak[lane] = fmaxf(peak[lane], peak[lane+n]);
        __syncthreads();
    }
    constexpr int bias = 1 << (Bits-1);
    const float scale = peak[0] / (bias-1);
    if (!lane) scales[row] = scale;
    for (int j = 2*lane; j < k; j += 512) {
        unsigned pair = 0;
        for (int t = 0; t < 2; ++t) {
            int q = scale ? __float2int_rn(expand(input[j+t]) / scale) : 0;
            pair |= unsigned(bias + q) << (Bits*t);
        }
#ifdef STRATA_USE_HIP
        size_t index = size_t(row) * (k/2) + j/2;
#else
        const int frag_lane = 4*(row%8) + (j%8)/2;
        size_t index = (((size_t(row/16)*(k/16) + j/16)*32 + frag_lane)*4
                        + 2*((j%16)/8) + (row%16)/8);
#endif
        if constexpr (Bits == 8) reinterpret_cast<uint16_t*>(packed)[index] = pair;
        else packed[index] = pair;
    }
}

#ifdef STRATA_USE_HIP
template<int Bits>
static __global__ void project_quant(const uint8_t* weights, const float* scales, const uint16_t* x,
                                     int n, int k, int rows, float* y, int ld) {
    using Half16 = _Float16 __attribute__((ext_vector_type(16)));
    using Float8 = float __attribute__((ext_vector_type(8)));
    const int lane = threadIdx.x % 32, l = lane % 16;
    const int col = blockIdx.x*64 + (threadIdx.x/32)*16 + l;
    const int token = blockIdx.y*16 + l;
    Float8 acc{};
    for (int j = 0; j < k; j += 16) {
        Half16 a{}, b{};
        if (token < rows) __builtin_memcpy(&a, x + size_t(token)*k + j, sizeof(a));
        if (col < n) {
            const auto* w = weights + (size_t(col)*k + j) / (8/Bits);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int u = (w[i/(8/Bits)] >> ((i%(8/Bits))*Bits)) & ((1<<Bits)-1);
                b[i] = _Float16(u - (1<<(Bits-1)));
            }
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    for (int i = 0; i < 8; ++i) {
        int t = blockIdx.y*16 + 2*i + lane/16;
        if (t < rows && col < n) y[size_t(t)*ld+col] = acc[i]*scales[col];
    }
}
#endif

static __global__ void bf16_to_half(uint16_t* x, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __half_as_ushort(__float2half(expand(x[i])));
}

static __global__ void gather(const float* taps, int ld, int first, uint16_t* packed) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width)
        packed[(blockIdx.y * 4 + blockIdx.z) * width + col] =
            bits(taps[(size_t(blockIdx.z) * ld + first + blockIdx.y) * width + col]);
}

struct ContextRows {
    int tap[64], position[64], slot[64];
};

static __global__ void gather_context(const float* taps, int ld, ContextRows rows, uint16_t* packed, int* pos) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    packed[(t * 4 + blockIdx.z) * width + c] = bits(taps[(blockIdx.z * ld + rows.tap[t]) * width + c]);
    if (!c && !blockIdx.z) pos[t] = rows.position[t];
}

static __global__ void scatter_context(const float* keys, const float* values, ContextRows rows, int cap,
                                       int cols, float* kc, float* vc) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t dest = (size_t(rows.slot[t]) * cap + rows.position[t] % cap) * cols + c;
    kc[dest] = keys[t * cols + c];
    vc[dest] = rounded(values[t * cols + c]);
}

// For residual updates vLLM normalizes with the unrounded sum of two BF16 operands,
// but uses the rounded residual for the numerator and carries it to the next layer.
static __global__ void norm(const float* x, float* residual, const float* weight, int cols, float* out) {
    __shared__ float sums[256];
    const int row = blockIdx.x, t = threadIdx.x;
    const size_t off = size_t(row) * cols;
    float sum = 0;
    for (int c = t; c < cols; c += 256) {
        float v = rounded(x[off+c]);
        if (residual) {
            v += rounded(residual[off+c]);
            residual[off+c] = rounded(v);
        }
        sum += v * v;
    }
    sums[t] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) sums[t] += sums[t+stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / cols + 1.e-6f);
    for (int c = t; c < cols; c += 256) {
        const float v = residual ? residual[off+c] : rounded(x[off+c]);
        out[off+c] = rounded(rounded(v * inv) * weight[c]);
    }
}

static __global__ void conv(const float* x, const float* dynamic, const uint16_t* base, int side, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= width) return;
    const float* d = dynamic + row * dyn_width + side * 2 * groups + c / 16;
    base += side * 2 * width + c;
    x += row * width + c;
    float y = (expand(base[0]) + rounded(d[0])) * rounded(x[0]);
    if (row % 8) y += (expand(base[width]) + rounded(d[groups])) * rounded(x[-width]);
    out[row * width + c] = rounded(y);
}

static __global__ void round_array(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = rounded(values[i]);
}

__device__ inline unsigned ordered(float x) {
    const unsigned b = bits(x);
    if ((b & 0x7fff) == 0) return 0x8000;
    return b & 0x8000 ? (b ^ 0xffff) : (b ^ 0x8000);
}

// Two byte histograms locate the 16th BF16 logit in each part. Keep all keys above
// the cutoff and fill ties in ascending token order.
static __global__ void radix_parts(const float* logits, int stride, int* ids) {
    __shared__ uint16_t keys[chunk];
    __shared__ unsigned hist[256];
    __shared__ int cutoff, need, count;
    const int lane = threadIdx.x, start = blockIdx.x * chunk;
    const int n = min(chunk, vocab - start);
    logits += size_t(blockIdx.y) * stride + start;
    ids += (blockIdx.y * parts + blockIdx.x) * top;
    for (int j = lane; j < n; j += 256) keys[j] = ordered(logits[j]);
    if (lane == 0) { cutoff = 0; need = top; count = 0; }
    __syncthreads();
    for (int shift = 8; shift >= 0; shift -= 8) {
        hist[lane] = 0;
        __syncthreads();
        for (int j = lane; j < n; j += 256)
            if (shift || (keys[j] & 0xff00) == cutoff)
                atomicAdd(&hist[(keys[j] >> shift) & 255], 1u);
        __syncthreads();
        if (!lane) {
            for (int b = 255; b >= 0; --b) {
                if (int(hist[b]) >= need) { cutoff |= b << shift; break; }
                need -= hist[b];
            }
        }
        __syncthreads();
    }
    for (int j = lane; j < n; j += 256)
        if (keys[j] > cutoff) ids[atomicAdd(&count, 1)] = start + j;
    __syncthreads();
    if (!lane)
        for (int j = 0; count < top; ++j)
            if (keys[j] == cutoff) ids[count++] = start + j;
}

static __global__ void merge_parts(const float* logits, int stride, const int* partial, int* ids) {
    __shared__ uint64_t best[128];
    const int row = blockIdx.x, lane = threadIdx.x;
    uint64_t ceiling = ~uint64_t(0);
    for (int k = 0; k < top; ++k) {
        uint64_t mine = 0;
        for (int i = lane; i < parts * top; i += 128) {
            const int id = partial[row * parts * top + i];
            const uint64_t key = (uint64_t(ordered(logits[size_t(row) * stride + id])) << 32) | ~uint32_t(id);
            if (key < ceiling && key > mine) mine = key;
        }
        best[lane] = mine;
        __syncthreads();
        for (int step = 64; step; step >>= 1) {
            if (lane < step && best[lane + step] > best[lane]) best[lane] = best[lane + step];
            __syncthreads();
        }
        ceiling = best[0];
        if (!lane) ids[row * top + k] = int(~uint32_t(ceiling));
        __syncthreads();
    }
}

__device__ inline int sample_candidate(const float* scores, const int* ids, int winner,
                                       double temperature, double uniform, DraftMass* mass) {
    // Sort by corrected score so a common uniform draw also couples
    // the batch scheduler's target and draft categorical samples.
    int order[top];
    double weights[top], total = 0;
    for (int i = 0; i < top; ++i) {
        weights[i] = exp(double(scores[i] - scores[winner]) / temperature);
        total += weights[i];
        int j = i;
        while (j && (scores[i] > scores[order[j-1]] ||
               (scores[i] == scores[order[j-1]] && ids[i] < ids[order[j-1]]))) {
            order[j] = order[j-1];
            --j;
        }
        order[j] = i;
    }
    double u = uniform * total;
    winner = order[top-1];
    for (int j = 0; j < top; ++j) {
        int i = order[j];
        if ((u -= weights[i]) < 0) { winner = i; break; }
    }
    for (int i = 0; i < top; ++i) {
        mass->ids[i] = ids[i];
        mass->probabilities[i] = weights[i] / total;
    }
    return winner;
}

// Conditional selector with the post-correction softcap, following qwen3_dflash2.py
// (vLLM contributors and Ciru, Apache-2.0). Sample only within the top-16 support.
template<bool Sample = false>
static __global__ void walk(const float* logits, int stride, const int* ids, const float* hidden,
                            const uint16_t* pred, const uint16_t* succ, int rows, int* tokens,
                            const Draws* draws = nullptr, DraftMass* mass = nullptr) {
    logits += size_t(blockIdx.x) * rows * stride;
    ids += blockIdx.x * rows * top;
    hidden += blockIdx.x * rows * rank;
    tokens += blockIdx.x * (rows + 1);
    __shared__ float scores[top];
    const int lane = threadIdx.x & 31, candidate = threadIdx.x >> 5;
    for (int r = 0; r < rows; ++r) {
        const int id = ids[r * top + candidate], prev = tokens[r];
        float edge = 0;
        for (int j = lane; j < rank; j += 32) {
            const float a = rounded(expand(pred[prev * rank + j]) * rounded(hidden[r * rank + j]));
            edge += a * expand(succ[id * rank + j]);
        }
        for (int s = 16; s; s >>= 1) edge += __shfl_down_sync(0xffffffffu, edge, s);
        if (!lane) scores[candidate] = 30.f * tanhf((rounded(logits[size_t(r)*stride + id]) + edge) / 30.f);
        __syncthreads();
        if (threadIdx.x == 0) {
            int winner = 0;
            for (int i = 1; i < top; ++i) if (scores[i] > scores[winner]) winner = i;
            if constexpr (Sample) {
                const auto& draw = draws[blockIdx.x];
                if (draw.temperature > 0) {
                    winner = sample_candidate(scores, ids + r*top, winner, draw.temperature, draw.uniform[r],
                                              mass + blockIdx.x*rows + r);
                }
            }
            tokens[r+1] = ids[r * top + winner];
        }
        __syncthreads();
    }
}
} // namespace ik2::draft::df2
