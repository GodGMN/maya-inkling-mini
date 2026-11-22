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

template <bool Prefill, bool Block = false, int Waves = 1, bool StageKV = (Waves > 1), bool Paged = false, class KV>
__global__ void rdna3_attn_part(const uint16_t* q, const float* r, const KV* kc, const KV* vc, int ring,
                               int H, int Hkv, const int* pos, int qpos, float qscale, const float* proj,
                               int dr, int extent, int window, float* scratch, int Tq, int q_ld, const float* bias, int r_ld = 0) {
    static_assert(Waves == 1 || Prefill);
    const int lane = threadIdx.x & 31, l = lane & 15, hi = lane >> 4;
    const int wave = threadIdx.x >> 5;
    const int split = blockIdx.y, splits = gridDim.y, tile = (blockIdx.z * Waves + wave) * 16;
    const int group = H / Hkv, kv_head = Prefill ? blockIdx.x / group : blockIdx.x;
    // In a verification block a column is a (token, head) pair sharing this KV head.
    const int queries = Prefill ? max(0, min(16, Tq - tile)) : Block ? min(16, Tq * group - tile) : group;
    const auto token = [&](int col) { return Prefill ? tile + col : Block ? (tile + col) / group : 0; };
    const auto head = [&](int col) { return Prefill ? (int)blockIdx.x : kv_head * group + (Block ? (tile + col) % group : col); };
    const int query_l = token(l), head_l = head(l);
    const int pos0 = pos ? *pos : qpos;
    const int end = pos0 + (Waves > 1 ? min(Tq, (int)(blockIdx.z + 1) * Waves * 16) : token(queries - 1) + 1);
    const int start = max(0, pos0 + (Waves > 1 ? (int)blockIdx.z * Waves * 16 : token(0)) + 1 - window);
    const size_t query_row = (size_t)query_l * H + head_l;
    const int kd = Hkv * 128;
    __shared__ AttnHalf16 query[Waves == 1 ? 16 : 1][8];
    __shared__ _Float16 values[StageKV ? 1 : 16][136];
    __shared__ AttnHalf16 keys[StageKV ? 16 : 1][8];
    __shared__ AttnHalf16 columns[StageKV ? 128 : 1];
    AttnHalf16 query_reg[Waves > 1 ? 8 : 1];
#pragma unroll
    for (int d = Waves == 1 ? hi : 0; d < 8; d += Waves == 1 ? 2 : 1) {
        AttnHalf16 fragment = {};
        if (l < queries)
            __builtin_memcpy(&fragment, q + (size_t)query_l * q_ld + head_l * 128 + d * 16, sizeof(fragment));
        if constexpr (Waves == 1) query[l][d] = fragment;
        else query_reg[d] = fragment;
    }
    if constexpr (Waves == 1) __syncthreads();
    float m = -INFINITY, sum = 0;
    AttnFloat8 out[8] = {};
    for (int key = (Paged ? start / 16 * 16 : start) + 16 * split; key < end; key += 16 * splits) {
        const int key_l = key + l;
        const size_t off = (size_t)(ring ? key_l % ring : key_l) * kd + kv_head * 128;
        const size_t page = Paged ? kv_page_base(key, kv_head, Hkv) : 0;
        AttnFloat8 dot = {};
        if constexpr (StageKV) {
            // Consecutive work-items load consecutive 16-feature fragments. All query waves share
            // these keys and the transposed values, including the final partial tile.
            for (int f = threadIdx.x; f < 128; f += Waves * 32) {
                const int t = f >> 3, d = f & 7, kp = key + t;
                const size_t src = Paged ? page + t * 128 + d * 16 :
                    (size_t)(ring ? kp % ring : kp) * kd + kv_head * 128 + d * 16;
                AttnHalf16 k = {};
                if (kp < end) k = cache_fragment(kc + src);
                keys[t][d] = k;
                if constexpr (Paged) {
                    AttnHalf16 v = cache_fragment(vc + page + f * 16);
                    if (key + 16 > end) {
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (key + i >= end) v[i] = 0;
                    }
                    columns[f] = v;
                } else {
                    AttnHalf16 v = {};
                    if (kp < end) v = cache_fragment(vc + src);
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        reinterpret_cast<_Float16*>(columns)[(d * 16 + i) * 16 + t] = v[i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if constexpr (!StageKV) {
                AttnHalf16 k = {};
                if (key_l < end) k = cache_fragment(kc + (Paged ? page + l * 128 : off) + d * 16);
                dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, Waves > 1 ? query_reg[d] : query[l][d], dot);
            } else {
                dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(keys[l][d], Waves > 1 ? query_reg[d] : query[l][d], dot);
            }
        }
        if constexpr (!StageKV && !Paged) {
#pragma unroll
            for (int d = hi; d < 8; d += 2) {
                AttnHalf16 v = {};
                if (key_l < end) v = cache_fragment(vc + off + d * 16);
                __builtin_memcpy(&values[l][d * 16], &v, sizeof(v));
            }
            __syncthreads();
        }
        float new_max = m;
        // A whole tile before the relative extent and inside the window needs only the scaled QK score.
        const int first_query = pos0 + token(0), last_query = pos0 + token(max(0, queries - 1));
        if (key + 15 <= first_query - extent && last_query - key < window) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot[i] *= qscale;
                new_max = fmaxf(new_max, dot[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int distance = pos0 + query_l - (key + 2 * i + hi);
                float v = -INFINITY;
                if (l < queries && distance >= 0 && distance < window) {
                    v = dot[i] * qscale;
                    if (distance < extent) {
                        if constexpr (Prefill) v += bias[query_row * extent + distance];
                        else for (int j = 0; j < dr; ++j) v += r[(size_t)query_l * r_ld + head_l * dr + j] * proj[distance * dr + j];
                    }
                }
                dot[i] = v;
                new_max = fmaxf(new_max, v);
            }
        }
        new_max = fmaxf(new_max, __shfl_xor(new_max, 16, 32));
        if (__any(new_max != m)) {
            const float rescale = m == -INFINITY ? 0.f : expf(m - new_max);
            sum *= rescale;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float factor = __shfl(rescale, 2 * i + hi, 32);
#pragma unroll
                for (int d = 0; d < 8; ++d) out[d][i] *= factor;
            }
        }
        m = new_max;
        AttnHalf16 prob;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = dot[i] == -INFINITY ? 0.f : expf(dot[i] - m);
            sum += v;
            const float peer = __shfl_xor(v, 16, 32);
            prob[2 * i] = (_Float16)(hi ? peer : v);
            prob[2 * i + 1] = (_Float16)(hi ? v : peer);
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            AttnHalf16 value;
            if constexpr (Paged && !StageKV) {
                value = cache_fragment(vc + page + (16 * d + l) * 16);
                if (key + 16 > end) {
#pragma unroll
                    for (int t = 0; t < 16; ++t) if (key + t >= end) value[t] = 0;
                }
            } else if constexpr (!StageKV) {
#pragma unroll
                for (int t = 0; t < 16; ++t) value[t] = values[t][16 * d + l];
            } else value = columns[16 * d + l];
            out[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(prob, value, out[d]);
        }
        if constexpr (!Paged || StageKV) __syncthreads();
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
            const size_t row = (size_t)token(dest) * H + head(dest);
            float* part = scratch + (row * splits + split) * 130;
#pragma unroll
            for (int d = 0; d < 8; ++d) part[2 + 16 * d + l] = out[d][i];
        }
    }
}

// Sixteen independent online softmax reductions per workgroup. Eight adjacent lanes cover a 128-wide
// head for one key. Each lane reads 16 consecutive KV elements without rounding the FP32 query.
template <class KV, bool Paged = false>
__global__ void rdna3_decode_split(const float* q, const float* r, const KV* kc, const KV* vc, int ring,
                                   int H, int Hkv, const int* position, float scale, const float* proj,
                                   int dr, int extent, int window, float* scratch) {
    constexpr int lanes = 8, streams = 128 / lanes;
    const int h = blockIdx.x, split = blockIdx.y, splits = gridDim.y, tid = threadIdx.x;
    const int stream = tid / lanes, channel = (tid % lanes) * 16;
    const int pos = *position, first = max(0, pos + 1 - window);
    const int span = (pos + 1 - first + splits - 1) / splits;
    const int start = first + split * span, end = min(pos + 1, start + span);
    const int kvh = h / (H / Hkv);
    float query[16], output[16] = {};
    __builtin_memcpy(query, q + h * 128 + channel, sizeof(query));
    float maximum = -INFINITY, sum = 0;
    for (int key = start + stream; key < end; key += streams) {
        const int slot = ring ? key % ring : key;
        const size_t page = Paged ? kv_page_base(slot, kvh, Hkv) : 0;
        const size_t off = (Paged ? page + (slot % 16) * 128 : ((size_t)slot * Hkv + kvh) * 128) + channel;
        const AttnHalf16 keys = cache_fragment(kc + off);
        float dot = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) dot += (float)keys[i] * query[i];
#pragma unroll
        for (int step = lanes / 2; step; step /= 2) dot += __shfl_xor(dot, step, lanes);
        float score = dot * scale;
        if (pos - key < extent) {
            const float* bias = proj + (size_t)(pos - key) * dr;
            float b = 0;
            for (int j = tid % lanes; j < dr; j += lanes) b += r[h * dr + j] * bias[j];
#pragma unroll
            for (int step = lanes / 2; step; step /= 2) b += __shfl_xor(b, step, lanes);
            score += b;
        }
        if constexpr (sizeof(KV) == 1) {
            // At magnitude 16 the FP32 score spacing reaches 2^-19. Refine these scores before softmax.
            if (fabsf(score) >= 16.f) {
                double precise = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) precise += (double)keys[i] * query[i];
                precise *= scale;
                if (pos - key < extent) {
                    const float* bias = proj + (size_t)(pos - key) * dr;
                    for (int j = tid % lanes; j < dr; j += lanes)
                        precise += (double)r[h * dr + j] * bias[j];
                }
#pragma unroll
                for (int step = lanes / 2; step; step /= 2) precise += __shfl_xor(precise, step, lanes);
                score = (float)precise;
            }
        }
        const float next = fmaxf(maximum, score);
        const float beta = expf(score - next), alpha = expf(maximum - next);
        sum = sum * alpha + beta;
        maximum = next;
        AttnHalf16 values;
        if constexpr (Paged) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                values[i] = __builtin_bit_cast(_Float16, e4m3fn_to_f16(vc[page + (channel + i) * 16 + slot % 16]));
        } else values = cache_fragment(vc + off);
#pragma unroll
        for (int i = 0; i < 16; ++i) output[i] = output[i] * alpha + (float)values[i] * beta;
    }
    __shared__ float maxes[streams], sums[streams], outputs[streams][128], factors[streams];
    if (tid % lanes == 0) {
        maxes[stream] = maximum;
        sums[stream] = sum;
    }
    __builtin_memcpy(&outputs[stream][channel], output, sizeof(output));
    __syncthreads();
    float* part = scratch + ((size_t)h * splits + split) * 130;
    if (tid < streams) {
        float m = maxes[tid];
#pragma unroll
        for (int step = streams / 2; step; step /= 2) m = fmaxf(m, __shfl_xor(m, step, streams));
        const float f = maxes[tid] == -INFINITY ? 0.f : expf(maxes[tid] - m);
        factors[tid] = f;
        float total = sums[tid] * f;
#pragma unroll
        for (int step = streams / 2; step; step /= 2) total += __shfl_xor(total, step, streams);
        if (tid == 0) { part[0] = m; part[1] = total; }
    }
    __syncthreads();
    float value = 0;
#pragma unroll
    for (int i = 0; i < streams; ++i) value += outputs[i][tid] * factors[i];
    part[2 + tid] = value;
}
