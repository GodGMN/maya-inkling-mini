// Included in the device pass, in the same namespace as grouped_expert_mma.
#pragma once

// Each group of four lanes owns a row. Each lane supplies eight K positions
// starting at 8*t, then eight more at 32+8*t. GGUF stores positions 0..7
// and 8..15 of each vector in the low and high nibbles of the same bytes.
__device__ __forceinline__ uint32_t bw_weight8(const uint8_t* block, int start) {
    const auto* words = reinterpret_cast<const uint32_t*>(block + 4 + (start / 16) * 8);
    const int shift = (start & 8) ? 4 : 0;
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint32_t n = (words[i] >> shift) & 0x0f0f0f0fu;
        n = (n | (n >> 4)) & 0x00ff00ffu;
        packed |= ((n | (n >> 8)) & 0xffffu) << (16 * i);
    }
    return packed;
}

struct BwFragment {
    uint32_t q[2], scales;
    float outer;
};

__device__ __forceinline__ BwFragment bw_quantize(const uint16_t* input) {
    const int t = threadIdx.x & 3;
    float x[2][8], peak[2];
    BwFragment frag{};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int start = j * 32 + 8 * t;
        float m = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[j][i] = h2f(input[start+i]);
            m = fmaxf(m, fabsf(x[j][i]));
        }
        peak[j] = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    }
    // A power of two per activation column keeps the UE4M3 scales in range
    // even for FP16 subnormals or values near 65504.
    float m = fmaxf(peak[0], peak[1]);
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float outer = frag.outer = m ? 0.25f * __uint_as_float(__float_as_uint(m) & 0x7f800000u) : 1.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float scale = peak[j] / (6.f * outer);
        uint16_t enc;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(enc) : "f"(scale));
        const uint32_t sf = max(1u, uint32_t(enc & 255));
        frag.scales |= __shfl_sync(0xffffffff, sf, 0, 4) << (16*j);
        frag.scales |= __shfl_sync(0xffffffff, sf, 2, 4) << (16*j+8);
        const float inv = 1.f / (2.f * ue4m3(sf) * outer);
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            uint16_t pair;
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %2, %1; mov.b16 %0, {q,0}; }"
                : "=h"(pair) : "f"(x[j][i] * inv), "f"(x[j][i+1] * inv));
            frag.q[j] |= uint32_t(pair) << (4*i);
        }
    }
    return frag;
}

__device__ __forceinline__ void bw_store(BlackwellActivation& dst, BwFragment x) {
    const int t = threadIdx.x & 3;
    dst.q[t] = x.q[0];
    dst.q[t+4] = x.q[1];
    if (!t) {
        dst.scales = x.scales;
        dst.outer = x.outer;
    }
}

__device__ __forceinline__ BwFragment bw_load(const BlackwellActivation& x) {
    const int t = threadIdx.x & 3;
    return {{x.q[t], x.q[t+4]}, x.scales, x.outer};
}

struct BwWeights {
    uint32_t q[4], scales;
    bool legacy;
};

__device__ __forceinline__ BwWeights bw_weights(const uint8_t* a, const uint8_t* b) {
    const int t = threadIdx.x & 3;
    const auto scales = *reinterpret_cast<const uint32_t*>((t & 1) ? b : a);
    BwWeights w;
    w.scales = (scales & 0x7f7f7f7fu) & ~__vcmpeq4(scales, 0x7f7f7f7fu);
    w.legacy = __vcmpeq4(scales, 0xffffffffu) != 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        w.q[2*j] = bw_weight8(a, 32*j + 8*t);
        w.q[2*j+1] = bw_weight8(b, 32*j + 8*t);
    }
    return w;
}

__device__ __forceinline__ void bw_nvfp4_dot(float (&sum)[4], const BwWeights& w, BwFragment x) {
    const int t = threadIdx.x & 3;
    float d[4] = {};
    const uint16_t zero = 0;
    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12,%12}, {%11}, {%12,%12};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(w.q[0]), "r"(w.q[1]), "r"(w.q[2]), "r"(w.q[3]), "r"(x.q[0]), "r"(x.q[1]), "r"(w.scales), "r"(x.scales), "h"(zero));
#pragma unroll
    for (int i = 0; i < 4; ++i)
        sum[i] += d[i] * __shfl_sync(0xffffffff, x.outer, 4 * (2*t + (i & 1)));
}
