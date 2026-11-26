// Each workgroup computes one rank of D. Only rank zero computes the two input norms.
__global__ void rdna3_gate_dots(LoopGateArgs a, int C, float* scratch) {
    constexpr int rank = 64;
    const int tid = threadIdx.x, r = blockIdx.x, j = blockIdx.y;
    const auto* weights = (const uint16_t*)a.D[j] + (size_t)r * 2 * C;
    const float* x = a.x[j] + blockIdx.z * C;
    const float* m = a.m[j] + blockIdx.z * C;
    float ax = 0.f, am = 0.f;
    for (int c = 4 * tid; c < C; c += 512) {
        const float4 xv = *(const float4*)(x + c), mv = *(const float4*)(m + c);
        const uint2 wx = *(const uint2*)(weights + c), wm = *(const uint2*)(weights + C + c);
        const uint32_t xwords[] = {wx.x, wx.y}, mwords[] = {wm.x, wm.y};
        const float xs[] = {xv.x, xv.y, xv.z, xv.w}, ms[] = {mv.x, mv.y, mv.z, mv.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int shift = (k & 1) * 16;
            ax += __uint_as_float((xwords[k / 2] >> shift) << 16) * xs[k];
            am += __uint_as_float((mwords[k / 2] >> shift) << 16) * ms[k];
        }
    }
    ax = wsum(ax);
    am = wsum(am);
    __shared__ float2 partials[4];
    if (!(tid & 31)) partials[tid / 32] = make_float2(ax, am);
    __syncthreads();
    float* out = scratch + (j * gridDim.z + blockIdx.z) * (2 * rank + 2);
    if (!tid) {
        for (int i = 1; i < 4; ++i) { ax += partials[i].x; am += partials[i].y; }
        out[2 * r] = ax;
        out[2 * r + 1] = am;
    }
    if (r == 0) {
        float sx = 0.f, sm = 0.f;
        for (int c = tid; c < C; c += 128) { sx += x[c] * x[c]; sm += m[c] * m[c]; }
        sx = wsum(sx);
        sm = wsum(sm);
        __syncthreads();
        if (!(tid & 31)) partials[tid / 32] = make_float2(sx, sm);
        __syncthreads();
        if (!tid) {
            for (int i = 1; i < 4; ++i) { sx += partials[i].x; sm += partials[i].y; }
            out[2 * rank] = sx;
            out[2 * rank + 1] = sm;
        }
    }
}

// Four groups each read 16 ranks at 64 consecutive channels. One group writes the gated result.
__global__ void rdna3_gate_apply(LoopGateArgs a, int C, float eps, const float* scratch) {
    constexpr int rank = 64, columns = 64, parts = 4;
    const int tid = threadIdx.x, column = tid % columns, part = tid / columns, j = blockIdx.y;
    const int c = blockIdx.x * columns + column;
    const float* src = scratch + (j * gridDim.z + blockIdx.z) * (2 * rank + 2);
    __shared__ float g[rank];
    __shared__ float2 partials[parts][columns];
    if (tid < rank) {
        const float rx = rsqrtf(src[2 * rank] / (float)C + eps), rm = rsqrtf(src[2 * rank + 1] / (float)C + eps);
        g[tid] = rx * src[2 * tid] + rm * src[2 * tid + 1];
    }
    __syncthreads();
    float forget = 0.f, gain = 0.f;
#pragma unroll
    for (int i = 0; i < rank / parts; ++i) {
        const int r = part * (rank / parts) + i;
        forget += gate_weight<true>(a.F[j], (size_t)r * C + c) * g[r];
        gain += gate_weight<true>(a.G[j], (size_t)r * C + c) * g[r];
    }
    partials[part][column] = make_float2(forget, gain);
    __syncthreads();
    if (!part) {
        for (int i = 1; i < parts; ++i) { forget += partials[i][column].x; gain += partials[i][column].y; }
        a.y[j][blockIdx.z * C + c] = (1.f - tanhf(forget)) * a.x[j][blockIdx.z * C + c] + tanhf(gain) * a.m[j][blockIdx.z * C + c];
    }
}

// Multi-token block-verify loop gates (T = 2..8): reads each BF16 weight ONCE across all T tokens.
template <int TFixed>
__global__ void rdna3_gate_dots_block(LoopGateArgs a, int n, int C, float* scratch) {
    constexpr int rank = 64;
    const int tid = threadIdx.x, r = blockIdx.x, j = blockIdx.y;
    const auto* weights = (const uint16_t*)a.D[j] + (size_t)r * 2 * C;
    const float* x = a.x[j];
    const float* m = a.m[j];
    float ax[TFixed] = {}, am[TFixed] = {};
    for (int c = 4 * tid; c < C; c += 512) {
        const uint2 wx = *(const uint2*)(weights + c), wm = *(const uint2*)(weights + C + c);
        const float wx0 = __uint_as_float((wx.x & 0xFFFFu) << 16), wx1 = __uint_as_float(wx.x & 0xFFFF0000u);
        const float wx2 = __uint_as_float((wx.y & 0xFFFFu) << 16), wx3 = __uint_as_float(wx.y & 0xFFFF0000u);
        const float wm0 = __uint_as_float((wm.x & 0xFFFFu) << 16), wm1 = __uint_as_float(wm.x & 0xFFFF0000u);
        const float wm2 = __uint_as_float((wm.y & 0xFFFFu) << 16), wm3 = __uint_as_float(wm.y & 0xFFFF0000u);
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            const float4 xv = *(const float4*)(x + (size_t)t * C + c);
            const float4 mv = *(const float4*)(m + (size_t)t * C + c);
            ax[t] = fmaf(wx0, xv.x, fmaf(wx1, xv.y, fmaf(wx2, xv.z, fmaf(wx3, xv.w, ax[t]))));
            am[t] = fmaf(wm0, mv.x, fmaf(wm1, mv.y, fmaf(wm2, mv.z, fmaf(wm3, mv.w, am[t]))));
        }
    }
    __shared__ float2 partials[4];
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        float vx = wsum(ax[t]), vm = wsum(am[t]);
        if (!(tid & 31)) partials[tid / 32] = make_float2(vx, vm);
        __syncthreads();
        if (!tid) {
            for (int i = 1; i < 4; ++i) { vx += partials[i].x; vm += partials[i].y; }
            float* out = scratch + ((size_t)t * n + j) * (2 * rank + 2);
            out[2 * r] = vx;
            out[2 * r + 1] = vm;
        }
        __syncthreads();
    }
    if (r == 0) {
        float sx[TFixed] = {}, sm[TFixed] = {};
        for (int c = tid; c < C; c += 128) {
#pragma unroll
            for (int t = 0; t < TFixed; ++t) {
                const float xv = x[(size_t)t * C + c], mv = m[(size_t)t * C + c];
                sx[t] += xv * xv;
                sm[t] += mv * mv;
            }
        }
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            float vx = wsum(sx[t]), vm = wsum(sm[t]);
            if (!(tid & 31)) partials[tid / 32] = make_float2(vx, vm);
            __syncthreads();
            if (!tid) {
                for (int i = 1; i < 4; ++i) { vx += partials[i].x; vm += partials[i].y; }
                float* out = scratch + ((size_t)t * n + j) * (2 * rank + 2);
                out[2 * rank] = vx;
                out[2 * rank + 1] = vm;
            }
            __syncthreads();
        }
    }
}

template <int TFixed>
__global__ void rdna3_gate_apply_block(LoopGateArgs a, int n, int C, float eps, const float* scratch) {
    constexpr int rank = 64, columns = 64, parts = 4;
    const int tid = threadIdx.x, column = tid % columns, part = tid / columns, j = blockIdx.y;
    const int c = blockIdx.x * columns + column;
    __shared__ float g[TFixed][rank];
    __shared__ float2 partials[parts][columns];
    for (int idx = tid; idx < TFixed * rank; idx += 256) {
        const int t = idx / rank, r = idx % rank;
        const float* src = scratch + ((size_t)t * n + j) * (2 * rank + 2);
        const float rx = rsqrtf(src[2 * rank] / (float)C + eps);
        const float rm = rsqrtf(src[2 * rank + 1] / (float)C + eps);
        g[t][r] = rx * src[2 * r] + rm * src[2 * r + 1];
    }
    __syncthreads();
    float forget[TFixed] = {}, gain[TFixed] = {};
#pragma unroll
    for (int i = 0; i < rank / parts; ++i) {
        const int r = part * (rank / parts) + i;
        const float fw = gate_weight<true>(a.F[j], (size_t)r * C + c);
        const float gw = gate_weight<true>(a.G[j], (size_t)r * C + c);
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            forget[t] = fmaf(fw, g[t][r], forget[t]);
            gain[t] = fmaf(gw, g[t][r], gain[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        partials[part][column] = make_float2(forget[t], gain[t]);
        __syncthreads();
        if (!part) {
            float f = forget[t], gn = gain[t];
            for (int i = 1; i < parts; ++i) { f += partials[i][column].x; gn += partials[i][column].y; }
            const size_t off = (size_t)t * C + c;
            a.y[j][off] = (1.f - tanhf(f)) * a.x[j][off] + tanhf(gn) * a.m[j][off];
        }
        __syncthreads();
    }
}
