// One wave reads eight 32-weight GGUF blocks per step and reuses its activation registers across output rows.
template <bool Gate, int Type, int Rows>
__global__ void rdna3_expert_token(const uint8_t* const* blobs, size_t offset, const int* tokens,
                                const float* x, const uint16_t* h, int K, int N, const float* gates,
                                const float* ups, float gate_cap, float up_cap, uint16_t* hidden, float* output) {
    const int slot = blockIdx.y, lane = threadIdx.x & 31, word = lane & 3;
    const int first = (blockIdx.x * 4 + threadIdx.x / 32) * Rows;
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
        const float a = wsum(sums[r][0]);
        if constexpr (Gate) {
            const float b = wsum(sums[r][1]);
            if (lane == 0) {
                const float g = a * (gates ? gates[slot] : 1.f), u = b * (ups ? ups[slot] : 1.f);
                hidden[(size_t)slot * N + first + r] =
                    f2h(gate_cap * tanhf(g / gate_cap) * (1.f / (1.f + expf(-g))) * up_cap * tanhf(u / up_cap));
            }
        } else if (lane == 0) {
            output[(size_t)slot * N + first + r] = a;
        }
    }
}

template <bool Gate, int Type, int Rows>
void launch_rdna3_expert(const uint8_t* const* blobs, int slots, size_t offset, const int* tokens,
                         const float* x, const uint16_t* h, int K, int N, const float* gates, const float* ups,
                         float gate_cap, float up_cap, uint16_t* hidden, float* output, cudaStream_t stream) {
    constexpr int rows_per_block = 4 * Rows;
    rdna3_expert_token<Gate, Type, Rows><<<dim3((N + rows_per_block - 1) / rows_per_block, slots), 128, 0, stream>>>(
        blobs, offset, tokens, x, h, K, N, gates, ups, gate_cap, up_cap, hidden, output);
}

template <bool Gate>
bool try_rdna3_expert(int type, const uint8_t* const* blobs, int slots, size_t offset, const int* tokens,
                      const float* x, const uint16_t* h, int K, int N, const float* gates, const float* ups,
                      float gate_cap, float up_cap, uint16_t* hidden, float* output, cudaStream_t stream) {
    static const bool iq4 = getenv("IK2_IQ4_MV") && atoi(getenv("IK2_IQ4_MV"));
    static const bool q8 = getenv("IK2_Q8_MV") && atoi(getenv("IK2_Q8_MV"));
    if (N % 2 || (Gate ? ((uintptr_t)x & 15) : ((uintptr_t)h & 7))) return false;
    if (type == kIQ4_NL && iq4)
        launch_rdna3_expert<Gate, kIQ4_NL, 2>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                             gate_cap, up_cap, hidden, output, stream);
    else if (type == kQ8_0 && q8)
        launch_rdna3_expert<Gate, kQ8_0, 2>(blobs, slots, offset, tokens, x, h, K, N, gates, ups,
                                           gate_cap, up_cap, hidden, output, stream);
    else return false;
    check(Gate ? "rdna3_expert_gate_up" : "rdna3_expert_down");
    return true;
}
