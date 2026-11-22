// One wave reads eight IQ4_NL blocks per step and reuses its activation registers across output rows.
template <bool Gate, int Rows>
__global__ void rdna3_iq4_token(const uint8_t* const* blobs, size_t offset, const int* tokens,
                                const float* x, const uint16_t* h, int K, int N, const float* gates,
                                const float* ups, float gate_cap, float up_cap, uint16_t* hidden, float* output) {
    const int slot = blockIdx.y, lane = threadIdx.x & 31, word = lane & 3;
    const int first = (blockIdx.x * 4 + threadIdx.x / 32) * Rows;
    if (first >= N) return;
    const size_t stride = (size_t)(K / 32) * 18;
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
                const uint8_t* block = weights + r * stride + b * 18 + m * offset;
                uint32_t packed;
                __builtin_memcpy(&packed, block + 2 + word * 4, sizeof(packed));
                const float dot = dot_u32_f4(iq4nl_lut4(packed, false), lo) +
                                  dot_u32_f4(iq4nl_lut4(packed, true), hi);
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
