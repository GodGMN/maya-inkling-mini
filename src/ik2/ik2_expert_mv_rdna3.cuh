// Recompute a projection when the wave sum loses twelve bits relative to a partial dot.
template <bool Padded>
__device__ __noinline__ float rdna3_iq4nl_precise_dot(const uint8_t* row, const float* x, int K) {
    const int lane = threadIdx.x & 31;
    double dot = 0;
    for (int b = 0; b < K / 32; ++b) {
        const uint8_t* block = row + b * 18;
        const int nibble = (block[2 + (lane & 15)] >> (lane < 16 ? 0 : 4)) & 15;
        const float w = h2f(*(const uint16_t*)block) * d_iq4nl[nibble];
        dot += (double)w * x[b * (Padded ? 36 : 32) + lane];
    }
    for (int d = 16; d; d /= 2) dot += __shfl_xor(dot, d, 32);
    return (float)dot;
}

template <bool Padded>
__device__ __forceinline__ float rdna3_iq4nl_gate_sum(float local, const uint8_t* row, const float* x, int K) {
    const float sum = wsum(local);
    if (__any(fabsf(local) > 4096.f * fabsf(sum))) return rdna3_iq4nl_precise_dot<Padded>(row, x, K);
    return sum;
}

// One wave reads eight 32-weight GGUF blocks per step and reuses its activation registers across output rows.
template <bool Gate, int Type, int Rows, int Waves, int FixedK>
__global__ void rdna3_expert_token(const uint8_t* const* blobs, size_t offset, const int* tokens,
                                const float* x, const uint16_t* h, int runtime_k, int N, const float* gates,
                                const float* ups, float gate_cap, float up_cap, uint16_t* hidden, float* output) {
    const int K = FixedK ? FixedK : runtime_k;
    const int slot = blockIdx.y, lane = threadIdx.x & 31, word = lane & 3;
    const int first = (blockIdx.x * Waves + threadIdx.x / 32) * Rows;
    if (first >= N) return;
    constexpr int bytes = Type == kIQ4_NL ? 18 : 34;
    const size_t stride = (size_t)(K / 32) * bytes;
    const uint8_t* weights = blobs[slot] + (Gate ? 0 : offset) + first * stride;
    const size_t input = (size_t)(Gate ? tokens[slot] : slot) * K;
    float sums[Rows][Gate ? 2 : 1] = {};
#pragma unroll 2
    for (int b = lane / 4; b < K / 32; b += 8) {
        const size_t k = input + b * 32 + 4 * word;
        float4 lo, hi;
        if constexpr (Gate) {
            lo = *(const float4*)(x + k);
            hi = *(const float4*)(x + k + 16);
        } else {
            const uint2 left = *(const uint2*)(h + k), right = *(const uint2*)(h + k + 16);
            const float2 a = u32_to_f2(left.x), b2 = u32_to_f2(left.y);
            const float2 c = u32_to_f2(right.x), d = u32_to_f2(right.y);
            lo = make_float4(a.x, a.y, b2.x, b2.y);
            hi = make_float4(c.x, c.y, d.x, d.y);
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int m = 0; m < (Gate ? 2 : 1); ++m) {
                const uint8_t* block = weights + r * stride + b * bytes + m * offset;
                uint32_t low, high;
                __builtin_memcpy(&low, block + 2 + word * 4, sizeof(low));
                if constexpr (Type == kIQ4_NL) {
                    high = iq4nl_lut4(low, true);
                    low = iq4nl_lut4(low, false);
                } else {
                    __builtin_memcpy(&high, block + 18 + word * 4, sizeof(high));
                }
                const float dot = dot_u32_f4(low, lo) + dot_u32_f4(high, hi);
                sums[r][m] += h2f(*(const uint16_t*)block) * dot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if constexpr (Gate) {
            float a, b;
            if constexpr (Type == kIQ4_NL) {
                a = rdna3_iq4nl_gate_sum<false>(sums[r][0], weights + r * stride, x + input, K);
                b = rdna3_iq4nl_gate_sum<false>(sums[r][1], weights + r * stride + offset, x + input, K);
            } else {
                a = wsum(sums[r][0]);
                b = wsum(sums[r][1]);
            }
            if (lane == 0) {
                const float g = a * (gates ? gates[slot] : 1.f), u = b * (ups ? ups[slot] : 1.f);
                hidden[(size_t)slot * N + first + r] =
                    f2h(gate_cap * tanhf(g / gate_cap) * (1.f / (1.f + expf(-g))) * up_cap * tanhf(u / up_cap));
            }
        } else {
            const float a = wsum(sums[r][0]);
            if (lane == 0) output[(size_t)slot * N + first + r] = a;
        }
    }
}

template <bool Gate, int Type, int Rows, int Waves, int FixedK>
void launch_rdna3_expert(const uint8_t* const* blobs, int slots, size_t offset, const int* tokens,
                         const float* x, const uint16_t* h, int K, int N, const float* gates, const float* ups,
                         float gate_cap, float up_cap, uint16_t* hidden, float* output, cudaStream_t stream) {
    constexpr int rows_per_block = Waves * Rows;
    rdna3_expert_token<Gate, Type, Rows, Waves, FixedK><<<dim3((N + rows_per_block - 1) / rows_per_block, slots), Waves * 32, 0, stream>>>(
        blobs, offset, tokens, x, h, K, N, gates, ups, gate_cap, up_cap, hidden, output);
}

template <bool Gate, int Type>
void dispatch_rdna3_expert(const uint8_t* const* blobs, int slots, size_t offset, const int* tokens,
                           const float* x, const uint16_t* h, int K, int N, const float* gates, const float* ups,
                           float gate_cap, float up_cap, uint16_t* hidden, float* output, cudaStream_t stream) {
    constexpr int width = Gate ? 2304 : 1152;
    constexpr int rows = Type == kIQ4_NL ? (Gate ? 2 : 4) : (Gate ? 1 : 2);
    constexpr int waves = Type == kIQ4_NL ? 2 : 4;
    if (K == width && N % rows == 0)
        launch_rdna3_expert<Gate, Type, rows, waves, width>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                                           gate_cap, up_cap, hidden, output, stream);
    else
        launch_rdna3_expert<Gate, Type, 2, 4, 0>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                                gate_cap, up_cap, hidden, output, stream);
}

template <bool Gate>
bool try_rdna3_expert(int type, const uint8_t* const* blobs, int slots, size_t offset, const int* tokens,
                      const float* x, const uint16_t* h, int K, int N, const float* gates, const float* ups,
                      float gate_cap, float up_cap, uint16_t* hidden, float* output, cudaStream_t stream) {
    static const bool iq4 = getenv("IK2_IQ4_MV") && atoi(getenv("IK2_IQ4_MV"));
    static const bool q8 = getenv("IK2_Q8_MV") && atoi(getenv("IK2_Q8_MV"));
    if (K % 32 || N % 2 || (Gate ? ((uintptr_t)x & 15) : ((uintptr_t)h & 7))) return false;
    if (type == kIQ4_NL && iq4)
        dispatch_rdna3_expert<Gate, kIQ4_NL>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                             gate_cap, up_cap, hidden, output, stream);
    else if (type == kQ8_0 && q8 && !(Gate && K == 32))
        dispatch_rdna3_expert<Gate, kQ8_0>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                           gate_cap, up_cap, hidden, output, stream);
    else return false;
    check(Gate ? "rdna3_expert_gate_up" : "rdna3_expert_down");
    return true;
}
