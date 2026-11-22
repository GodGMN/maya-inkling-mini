// One wave streams tiles of 16 keys against a KV head’s queries. Decode packs heads, prefill packs tokens.
// WMMA computes QK and PV. The softmax statistics, relative bias and partial outputs are FP32.
using AttnHalf16 = _Float16 __attribute__((ext_vector_type(16)));
using AttnFloat8 = float __attribute__((ext_vector_type(8)));

template <class KV>
__device__ __forceinline__ AttnHalf16 cache_fragment(const KV* ptr) {
    AttnHalf16 out;
    if constexpr (sizeof(KV) == 2) {
        __builtin_memcpy(&out, ptr, sizeof(out));
    } else {
        using BytePairs = uint16_t __attribute__((ext_vector_type(8)));
        using HalfPairs = uint32_t __attribute__((ext_vector_type(8)));
        BytePairs bytes;
        HalfPairs halves;
        __builtin_memcpy(&bytes, ptr, sizeof(bytes));
#pragma unroll
        for (int i = 0; i < 8; ++i) halves[i] = e4m3fnx2_to_h2_u32(bytes[i]);
        out = __builtin_bit_cast(AttnHalf16, halves);
    }
    return out;
}

template <bool Prefill, class KV>
__global__ void rdna3_attn_part(const uint16_t* q, const float* r, const KV* kc, const KV* vc, int ring,
                               int H, int Hkv, const int* pos, int qpos, float qscale, const float* proj,
                               int dr, int extent, int window, float* scratch, int Tq, int q_ld, const float* bias) {
    const int lane = threadIdx.x, l = lane & 15, hi = lane >> 4;
    const int split = blockIdx.y, splits = gridDim.y, tile = blockIdx.z * 16;
    const int group = H / Hkv, kv_head = Prefill ? blockIdx.x / group : blockIdx.x;
    const int queries = Prefill ? min(16, Tq - tile) : group;
    const int head_l = Prefill ? blockIdx.x : kv_head * group + l;
    const int query_l = Prefill ? tile + l : 0;
    const int pos0 = pos ? *pos : qpos;
    const int end = pos0 + (Prefill ? tile + queries : 1);
    const int start = max(0, pos0 + (Prefill ? tile : 0) + 1 - window);
    const size_t query_row = (size_t)query_l * H + head_l;
    const int kd = Hkv * 128;
    __shared__ AttnHalf16 query[16][8];
    __shared__ _Float16 values[16][136];
#pragma unroll
    for (int d = hi; d < 8; d += 2) {
        AttnHalf16 fragment = {};
        if (l < queries)
            __builtin_memcpy(&fragment, q + (size_t)query_l * q_ld + head_l * 128 + d * 16, sizeof(fragment));
        query[l][d] = fragment;
    }
    __syncthreads();
    float m = -INFINITY, sum = 0;
    AttnFloat8 out[8] = {};
    for (int key = start + 16 * split; key < end; key += 16 * splits) {
        const int key_l = key + l;
        const size_t off = (size_t)(ring ? key_l % ring : key_l) * kd + kv_head * 128;
        AttnFloat8 dot = {};
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            AttnHalf16 k = {};
            if (key_l < end) k = cache_fragment(kc + off + d * 16);
            dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, query[l][d], dot);
        }
#pragma unroll
        for (int d = hi; d < 8; d += 2) {
            AttnHalf16 v = {};
            if (key_l < end) v = cache_fragment(vc + off + d * 16);
            __builtin_memcpy(&values[l][d * 16], &v, sizeof(v));
        }
        __syncthreads();
        float new_max = m;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int distance = pos0 + query_l - (key + 2 * i + hi);
            float v = -INFINITY;
            if (l < queries && distance >= 0 && distance < window) {
                v = dot[i] * qscale;
                if (distance < extent) {
                    if constexpr (Prefill) v += bias[query_row * extent + distance];
                    else for (int j = 0; j < dr; ++j) v += r[head_l * dr + j] * proj[distance * dr + j];
                }
            }
            dot[i] = v;
            new_max = fmaxf(new_max, v);
        }
        new_max = fmaxf(new_max, __shfl_xor(new_max, 16, 32));
        const float rescale = m == -INFINITY ? 0.0f : expf(m - new_max);
        sum *= rescale;
        m = new_max;
        AttnHalf16 prob;
        float row_scale[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = dot[i] == -INFINITY ? 0.0f : expf(dot[i] - m);
            sum += v;
            prob[2 * i] = (_Float16)__shfl(v, l, 32);
            prob[2 * i + 1] = (_Float16)__shfl(v, l + 16, 32);
            row_scale[i] = __shfl(rescale, 2 * i + hi, 32);
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            AttnHalf16 value;
#pragma unroll
            for (int t = 0; t < 16; ++t) value[t] = values[t][16 * d + l];
#pragma unroll
            for (int i = 0; i < 8; ++i) out[d][i] *= row_scale[i];
            out[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(prob, value, out[d]);
        }
        __syncthreads();
    }
    sum += __shfl_xor(sum, 16, 32);
    if (lane < queries) {
        float* part = scratch + (query_row * splits + split) * 130;
        part[0] = m;
        part[1] = sum;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int dest = 2 * i + hi;
        if (dest < queries) {
            const size_t row = Prefill ? (size_t)(tile + dest) * H + blockIdx.x : kv_head * group + dest;
            float* part = scratch + (row * splits + split) * 130;
#pragma unroll
            for (int d = 0; d < 8; ++d) part[2 + 16 * d + l] = out[d][i];
        }
    }
}
