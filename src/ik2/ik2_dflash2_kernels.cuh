// Inkling-2-Mini DFlash2. All GEMM outputs enter the next operator rounded to BF16.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace ik2::draft::df2 {
constexpr int width = 2304, groups = width / 16, dyn_width = 4 * groups;
constexpr int vocab = 200058, codebook_rows = 201024, rank = 256, top = 16, chunk = 4096;
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

// Each output uses four K partitions and the same ordered MMA reductions for every batch size.
// A warp keeps its weight fragment in registers across all of the tile's 8-row input panels.
template <int Panels, bool Half = false>
static __global__ void project(const uint16_t* weight, const uint16_t* x, int n, int k, int rows,
                               float* y, int ld) {
    const int lane = threadIdx.x % 32, split = threadIdx.x / 32;
    const int g = lane / 4, pair = 2 * (lane % 4), col = blockIdx.x * 16 + g;
    const int first = blockIdx.y * Panels * 8;
    float acc[Panels][4]{};
    for (int i = 16 * split; i < k; i += 64) {
        uint4 a;
        if constexpr (Half) {
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
            if constexpr (Half) { IK2_DRAFT_MMA("f16"); }
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
            const int t = first + 8*p + pair;
            if (t < rows && col < n) y[size_t(t) * ld + col] = v.x;
            if (t+1 < rows && col < n) y[size_t(t+1) * ld + col] = v.y;
            if (t < rows && col+8 < n) y[size_t(t) * ld + col+8] = v.z;
            if (t+1 < rows && col+8 < n) y[size_t(t+1) * ld + col+8] = v.w;
        }
        __syncthreads();
    }
}

template <bool Half = false>
static void project_rows(const uint16_t* weight, const uint16_t* x, int n, int k, int rows, float* y,
                         int ld, cudaStream_t stream) {
    const int blocks = (n + 15) / 16;
    if (rows <= 8) project<1, Half><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld);
    else if (rows <= 16) project<2, Half><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld);
    else if (rows <= 32) project<4, Half><<<blocks, 128, 0, stream>>>(weight, x, n, k, rows, y, ld);
    else project<8, Half><<<dim3(blocks, (rows+63)/64), 128, 0, stream>>>(weight, x, n, k, rows, y, ld);
}
#endif

__device__ inline uint16_t bits(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
__device__ inline float expand(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ inline float rounded(float x) { return expand(bits(x)); }

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

// One warp per candidate, with the winning token fed into the next row.
static __global__ void walk(const float* logits, int stride, const int* ids, const float* hidden,
                            const uint16_t* pred, const uint16_t* succ, int rows, int* tokens) {
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
            tokens[r+1] = ids[r * top + winner];
        }
        __syncthreads();
    }
}
} // namespace ik2::draft::df2
