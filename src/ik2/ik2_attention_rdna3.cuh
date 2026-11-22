// One wave streams tiles of 16 keys against a KV head’s queries. Decode packs heads, prefill packs tokens.
// WMMA computes QK and PV. The softmax statistics, relative bias and partial outputs are FP32.
using AttnHalf16 = _Float16 __attribute__((ext_vector_type(16)));
using AttnFloat8 = float __attribute__((ext_vector_type(8)));

template <class KV>
__device__ __forceinline__ _Float16 cache_half(KV x) {
    if constexpr (sizeof(KV) == 1) return __builtin_bit_cast(_Float16, e4m3fn_to_f16(x));
    else return __builtin_bit_cast(_Float16, x);
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
    AttnHalf16 query[8] = {};
    if (l < queries) {
#pragma unroll
        for (int d = 0; d < 8; ++d)
            __builtin_memcpy(query + d, q + (size_t)query_l * q_ld + head_l * 128 + d * 16, sizeof(AttnHalf16));
    }
    float m = -INFINITY, sum = 0;
    AttnFloat8 out[8] = {};
    for (int key = start + 16 * split; key < end; key += 16 * splits) {
        const int key_l = key + l;
        const size_t off = (size_t)(ring ? key_l % ring : key_l) * kd + kv_head * 128;
        AttnFloat8 dot = {};
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            AttnHalf16 k = {};
            if (key_l < end) {
#pragma unroll
                for (int i = 0; i < 16; ++i) k[i] = cache_half(kc[off + d * 16 + i]);
            }
            dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, query[d], dot);
        }
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
            for (int t = 0; t < 16; ++t) {
                const int kp = key + t;
                const size_t row = (size_t)(ring ? kp % ring : kp) * kd + kv_head * 128;
                value[t] = kp < end ? cache_half(vc[row + 16 * d + l]) : (_Float16)0;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) out[d][i] *= row_scale[i];
            out[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(prob, value, out[d]);
        }
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
