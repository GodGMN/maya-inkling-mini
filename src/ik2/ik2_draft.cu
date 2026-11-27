// src/ik2/ik2_draft.cu - tintero's drafter (the design: ik2_draft.hpp).
#include "ik2_draft.hpp"
#include "ik2_dflash2_kernels.cuh"
#include "ik2_draft_weights.hpp"
#include "ik2_kernels.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ik2::draft {
namespace {

namespace k = ik2::k;

void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("drafter: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("drafter: ") + what + ": cuBLAS " +
                                                             std::to_string((int) e));
}
template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    cu(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
    return p;
}
float bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

// BF16 or packed integers with one scale per output channel.
struct Backbone { uint16_t* data; float* scales; };

struct DLayer {
    struct Conv { uint16_t *projection, *base; };
    Conv attention_conv{}, mlp_conv{};
    bool local = false;
    float *in_norm, *post_norm, *dflare, *q_norm, *k_norm, *k_norm_ctx;
    Backbone wq, wk, wv, wo, wgu, wd;
    uint16_t *wkc, *wvc;   // DSpark context projections, BF16 [out][in]
    float *kc = nullptr, *vc = nullptr;                    // the context caches, F32 [cap][KV * 128] (a window's
    uint16_t *kc16 = nullptr, *vc16 = nullptr;             //  layer) or FP16 (the layer that sees the whole context)
    int cap = 0;                                           // their rows: a ring of the last cap positions
};

struct Drafter {
    struct DFlash2 {
        uint16_t *fc, *selector, *pred, *succ, *packed;
        float *hidden_norm, *dynamic;
        Draws *draws, *host_draws;
        DraftMass *mass, *host_mass;
        bool sample;
        int *parts, *ids;
        const uint16_t* head = nullptr;
    };
    DFlash2* flash = nullptr;
    int bits = 16;
    int taps = 8, C = 4608, H = 48, KV = 8, F = 18432, R = 256, V = 0, ctx = 0;
    int cap = 0;   // the rows of the layer without a window's context cache (IK2_DRAFT_CTX, default the context)
    float* ctx_tmp = nullptr;   // write_context's K and V rows before an FP16 cache, [2][kMixRows][KV * 128]
    float eps = 1e-6f, theta = 1e6f;
    std::vector<DLayer> L;
    float *alpha, *norm, *mask, *mk_dflare, *rnn_b;
    uint16_t *w1, *rnn_w;
    std::vector<uint16_t> hout_w, hout_b;   // host BF16, for the fold
    uint16_t *fold_w = nullptr, *fold_b = nullptr;   // FP16 [R][n_rows], [n_rows]
    int n_rows = 0;
    const int8_t* head_q8 = nullptr;      // the target's head as planar Q8_0 (set_head_q8): the base logits through it
    const uint16_t* head_q8d = nullptr;
    int* shortlist = nullptr;
    float* shortlist_scores = nullptr;
    cublasHandle_t blas;
    cudaStream_t s;
    size_t vram = 0;
    // work
    float *x, *res, *h, *q, *kk, *vv, *o, *gu, *f_rnn, *base, *state, *latent, *proj, *scratch, *mix;
    uint16_t *hb, *hb2, *rnn_in;
    __half* x16;
    int *pos, *ctx_pos, *toks;
    int* h_out = nullptr;   // pinned proposal tokens, including anchors in batched mode
    int* h_meta = nullptr;  // {P, anchor} per active slot, followed by slot IDs
    int* d_meta = nullptr;  // device copy of h_meta
    struct PropGraph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        const float* anchor_h = nullptr;
        int n_rows = 0, n_vocab = 0;
    };
    struct SlotGraphs { PropGraph width[kBlock][5]; };
    std::vector<SlotGraphs> graphs;
    PropGraph batch_graph[kBlock + 1][kBlock + 1][2];  // [active][verify width]
    int slot = 0;
};

namespace {

constexpr int kMixRows = 512;
constexpr int kFoldChunk = 4096;

__global__ void iota_kernel(int* out, int start, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = start + i;
}

__global__ void init_block_kernel(const float* anchor_h, const float* mask, float* res, int* pos, int* toks, int T,
                                  int C, int P, int anchor) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && blockIdx.x == 0) {
        if (threadIdx.x < T) pos[threadIdx.x] = P + threadIdx.x;
        if (threadIdx.x == 0) toks[0] = anchor;
    }
    if (c < C) res[(size_t) t * C + c] = (t == 0 ? anchor_h : mask)[c];
}

__global__ void init_block_kernel_d(const float* anchor_h, const float* mask, float* res, int* pos, int* toks, int T,
                                    int C, const int* meta, int window = kBlock) {
    const int batch = blockIdx.z;
    anchor_h += batch * C;
    res += batch * T * C;
    pos += batch * T;
    toks += batch * window;
    meta += 2 * batch;
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && blockIdx.x == 0) {
        if (threadIdx.x < T) pos[threadIdx.x] = meta[0] + threadIdx.x;
        if (threadIdx.x == 0) toks[0] = meta[1];
    }
    if (c < C) res[(size_t) t * C + c] = (t == 0 ? anchor_h : mask)[c];
}

__global__ void f16_to_f32_kernel(const uint16_t* src, size_t n, float* dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(__ushort_as_half(src[i]));
}

__global__ void pack_f16_strided_kernel(const float* src, int m, int n_rows, int r0, uint16_t* dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i < m) dst[(size_t) r * n_rows + r0 + i] = __half_as_ushort(__float2half_rn(src[(size_t) r * m + i]));
}

// Each warp keeps the strongest base candidate among 32 vocabulary entries.
// Only the drafter uses this shortlist; the target still verifies the full vocabulary.
__global__ void draft_shortlist(const float* base, int ld, int vocab, int stride, int* picks) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    float val = token < vocab ? base[(size_t)row * ld + token] : -INFINITY;
    int idx = token;
    for (int step = 16; step; step >>= 1) {
        float v = __shfl_down_sync(0xffffffff, val, step);
        int j = __shfl_down_sync(0xffffffff, idx, step);
        if (v > val || (v == val && j < idx)) val = v, idx = j;
    }
    if (!(threadIdx.x & 31) && token < vocab) picks[row * stride + token / 32] = idx;
}
__global__ void fold_vocab_rows(const float* src, int m, int r0, int R, uint16_t* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * R) {
        const int i = idx / R, r = idx % R;
        dst[(size_t)(r0 + i) * R + r] = __half_as_ushort(__float2half_rn(src[(size_t)r * m + i]));
    }
}
__global__ void shortlist_gemv(const int* ids, int n, const uint16_t* W, const float* x, const float* base,
                               const uint16_t* bias, int R, float* scores) {
    const int lane = threadIdx.x & 31;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i >= n) return;
    const int token = ids[i];
    float dot = 0.0f;
    for (int r = lane; r < R; r += 32)
        dot = fmaf(__half2float(__ushort_as_half(W[(size_t)token * R + r])), x[r], dot);
    for (int step = 16; step; step >>= 1) dot += __shfl_down_sync(0xffffffff, dot, step);
    if (lane == 0) scores[i] = dot + base[token] + __half2float(__ushort_as_half(bias[token]));
}
__global__ void shortlist_pick(int* out, const int* ids) { *out = ids[*out]; }

uint16_t* up16(Drafter* d, const std::vector<uint16_t>& v) {
    uint16_t* p = dalloc<uint16_t>(v.size());
    cu(cudaMemcpy(p, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "upload");
    d->vram += v.size() * 2;
    return p;
}
uint16_t* projection(Drafter* d, const std::vector<uint16_t>& values, int k) {
    auto* weights = up16(d, values);
#ifndef STRATA_USE_HIP
    if (d->flash) {
        auto* packed = dalloc<uint16_t>(values.size());
        const int n = values.size() / k;
        df2::pack_projection<<<dim3(n / 16, (k + 63) / 64), 128, 0, d->s>>>(
            weights, k, reinterpret_cast<uint4*>(packed));
        cu(cudaFree(weights), "pack draft projection");
        return packed;
    }
#endif
    return weights;
}
Backbone backbone(Drafter* d, const std::vector<uint16_t>& values, int k) {
    if (d->bits == 16) return {projection(d, values, k), nullptr};
    const int n = values.size() / k;
    auto* input = up16(d, values);
    auto* packed = dalloc<uint16_t>(values.size() * d->bits / 16);
    auto* scales = dalloc<float>(n);
    if (d->bits == 8)
        df2::quantize_rows<8><<<n, 256, 0, d->s>>>(input, k, reinterpret_cast<uint8_t*>(packed), scales);
    else
        df2::quantize_rows<4><<<n, 256, 0, d->s>>>(input, k, reinterpret_cast<uint8_t*>(packed), scales);
    cu(cudaFree(input), "quantize draft projection");
    d->vram -= values.size()*2;
    d->vram += values.size()*d->bits/8 + n*sizeof(float);
    return {packed, scales};
}

float* up32(Drafter* d, const std::vector<uint16_t>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = bf(v[i]);
    float* p = dalloc<float>(f.size());
    cu(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
    d->vram += f.size() * 4;
    return p;
}

// Y [T][N] (ldy) = X [T][K] BF16 . W [N][K]^T BF16, F32 sums
void gemm_bf(Drafter* d, const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
#ifndef STRATA_USE_HIP
    if (d->flash) return df2::project_rows(W, X, N, K, T, Y, ldy, d->s);
#endif
#ifdef STRATA_USE_HIP
    static const bool bf16_mv = !getenv("IK2_DRAFT_BF16_MV") || atoi(getenv("IK2_DRAFT_BF16_MV"));
    const int mv_rows = d->flash ? kBlock * kBlock : kBlock;
    if (bf16_mv && T >= 1 && T <= mv_rows && K % 128 == 0 && ((uintptr_t)W & 7) == 0 && ((uintptr_t)X & 7) == 0) {
        for (int t = 0; t < T; t += kBlock)
            k::rdna3_bf16_gemm(W, N, K, X + t*K, std::min(kBlock, T-t), Y + t*ldy, ldy, d->s);
        return;
    }
#endif
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_bf");
}
void project_backbone(Drafter* d, Backbone w, int n, int kdim, const uint16_t* x, int rows, float* y, int ld) {
    if (d->bits == 16) return gemm_bf(d, w.data, n, kdim, x, rows, y, ld);
#ifndef STRATA_USE_HIP
    if (d->bits == 8) df2::project_rows<false, 8>(w.data, x, n, kdim, rows, y, ld, d->s, w.scales);
    else df2::project_rows<false, 4>(w.data, x, n, kdim, rows, y, ld, d->s, w.scales);
#else
    dim3 grid((n+63)/64, (rows+15)/16);
    auto* weights = reinterpret_cast<const uint8_t*>(w.data);
    if (d->bits == 8) df2::project_quant<8><<<grid, 128, 0, d->s>>>(weights, w.scales, x, n, kdim, rows, y, ld);
    else df2::project_quant<4><<<grid, 128, 0, d->s>>>(weights, w.scales, x, n, kdim, rows, y, ld);
#endif
}

void backbone_input(Drafter* d, const float* x, int count, uint16_t* out) {
    if (d->bits == 16) k::to_bf16(x, count, out, d->s);
    else k::to_f16(x, count, out, d->s);
}

void gemm_h(Drafter* d, const uint16_t* W, int N, int K, const void* X, int T, float* Y, int ldy) {
#ifndef STRATA_USE_HIP
    if (d->flash) return df2::project_rows<true>(W, static_cast<const uint16_t*>(X), N, K, T, Y, ldy, d->s);
#endif
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_h");
}

void layer_attn(Drafter* d, DLayer& l, int P, float* out, int graph_splits = 0) {
    const int T = kBlock, HD = d->H * 128, KD = d->KV * 128;
    k::to_bf16(d->h, (size_t) T * d->C, d->hb, d->s);
    project_backbone(d, l.wq, HD, d->C, d->hb, T, d->q, HD);
    project_backbone(d, l.wk, KD, d->C, d->hb, T, d->kk, KD);
    project_backbone(d, l.wv, KD, d->C, d->hb, T, d->vv, KD);
    k::rms(d->q, l.q_norm, T * d->H, 128, d->eps, d->q, d->s);
    k::rms(d->kk, l.k_norm, T * d->KV, 128, d->eps, d->kk, d->s);
    k::rope_neox(d->q, T, d->H, d->pos, d->theta, d->s);
    k::rope_neox(d->kk, T, d->KV, d->pos, d->theta, d->s);
    // Local layers (0, 1, 3, 4) match SGLang inkling2_mtp.py's window_anchor_at_start=True with window=1024:
    // every query in the block [P .. P+7] sees context [max(0, P - 1023) .. P-1] plus all 8 block tokens.
    if (graph_splits > 0) {
        const int window = l.local ? 1023 : l.cap;
        const int S = l.local ? 4 : graph_splits;
        if (l.kc16)
            k::draft_attn_f16_graph(d->q, l.kc16, l.vc16, d->d_meta, window, d->kk, d->vv, T, d->H, d->KV,
                                    1.0f / std::sqrt(128.0f), S, d->scratch, d->o, d->s, l.cap);
        else
            k::draft_attn_graph(d->q, l.kc, l.vc, d->d_meta, window, d->kk, d->vv, T, d->H, d->KV,
                                1.0f / std::sqrt(128.0f), S, d->scratch, d->o, d->s, l.cap);
    } else {
        const int k_lo = l.local ? std::max(0, P - 1023) : std::max(0, P - l.cap);
        if (l.kc16)
            k::draft_attn_f16(d->q, l.kc16, l.vc16, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f),
                              d->scratch, d->o, d->s, l.cap);
        else
            k::draft_attn(d->q, l.kc, l.vc, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f), d->scratch,
                          d->o, d->s, l.cap);
    }
    k::to_bf16(d->o, (size_t) T * HD, d->hb, d->s);
    project_backbone(d, l.wo, d->C, HD, d->hb, T, out, d->C);
}

void layer_mlp(Drafter* d, DLayer& l, float* out) {
    const int T = kBlock;
    k::to_bf16(d->h, (size_t) T * d->C, d->hb, d->s);
    project_backbone(d, l.wgu, 2 * d->F, d->C, d->hb, T, d->gu, 2 * d->F);
    k::silu_mul_bf16(d->gu, T, d->F, d->hb2, d->s);
    project_backbone(d, l.wd, d->C, d->F, d->hb2, T, out, d->C);
}

}  // namespace

const std::vector<int>& tap_layers() {
    static const std::vector<int> v = {5, 11, 17, 23, 29, 35, 41};
    return v;
}

Drafter* load(const std::string& path, int C, int ctx, cublasHandle_t blas, cudaStream_t s, int slots) {
    const StFile f = st_open(path);
    auto* d = new Drafter;
    d->graphs.resize(slots);
    d->C = C, d->ctx = ctx, d->blas = blas, d->s = s;
    {
        // (IK2_DRAFT_CTX: the positions the layer without a window sees, the last ones; default the whole context)
        const int want = getenv("IK2_DRAFT_CTX") ? std::atoi(getenv("IK2_DRAFT_CTX")) : ctx;
        d->cap = std::min(ctx, std::max(want, 2048));
    }
    const bool mini = C == 2304;
    if (mini) {
        d->H = 24;
        d->F = 9216;
        d->taps = 4;
    } else if (C != 4608) {
        throw std::runtime_error("drafter: unsupported target width");
    }
    if (f.t.count("candidate_selector.hidden_projection.weight")) {
        if (!mini) throw std::runtime_error("DFlash2 requires Inkling-2-Mini");
        d->flash = new Drafter::DFlash2;
        const char* quant = std::getenv("IK2_DFLASH2_QUANT");
        if (!quant || !std::strcmp(quant, "q8")) d->bits = 8;
        else if (!std::strcmp(quant, "bf16")) d->bits = 16;
        else if (!std::strcmp(quant, "q4")) d->bits = 4;
        else throw std::runtime_error("IK2_DFLASH2_QUANT expects bf16, q8 or q4");
        const char* sample = std::getenv("IK2_DFLASH2_SAMPLE");
        d->flash->sample = !sample || !std::strcmp(sample, "prob");
        if (!d->flash->sample && std::strcmp(sample, "greedy"))
            throw std::runtime_error("IK2_DFLASH2_SAMPLE expects prob or greedy");
        d->H = 32;
        d->F = 6912;
        d->theta = 10000.f;
    }
    const int layers = mini && !d->flash ? 3 : 5;
    const int64_t H = d->H, KV = d->KV, F = d->F, R = d->R;
    if (!d->flash) {
        auto it = f.t.find("markov_head.markov_w1.weight");
        if (it == f.t.end() || it->second.shape.size() != 2) throw std::runtime_error("drafter: no markov_w1");
        d->V = (int) it->second.shape[0];
        // DSpark mixes taps separately for each layer.
        const auto fw = st_bf16(f, "fuse_weight", {layers + 1, d->taps});
        std::vector<float> a(fw.size());
        for (int r = 0; r <= layers; ++r) {
            float m = -1e30f, sum = 0;
            for (int t = 0; t < d->taps; ++t) m = std::max(m, bf(fw[r * d->taps + t]));
            for (int t = 0; t < d->taps; ++t) sum += (a[r * d->taps + t] = std::exp(bf(fw[r * d->taps + t]) - m));
            for (int t = 0; t < d->taps; ++t) a[r * d->taps + t] /= sum;
        }
        d->alpha = dalloc<float>(a.size());
        cu(cudaMemcpy(d->alpha, a.data(), a.size() * 4, cudaMemcpyHostToDevice), "alpha");
    }
    const int local_ids[] = {0, 1, 3, 4};
    d->L.resize(layers);
    for (int i = 0; i < layers; ++i) {
        DLayer& l = d->L[i];
        const std::string b = "layers." + std::to_string(i) + ".";
        l.local = d->flash || (mini ? i != 1 : std::find(std::begin(local_ids), std::end(local_ids), i) != std::end(local_ids));
        l.in_norm = up32(d, st_bf16(f, b + "input_layernorm.weight", {C}));
        l.post_norm = up32(d, st_bf16(f, b + "post_attention_layernorm.weight", {C}));
        if (!d->flash) l.dflare = up32(d, st_bf16(f, b + "dflare_norm.weight", {C}));
        l.q_norm = up32(d, st_bf16(f, b + "self_attn.q_norm.weight", {128}));
        l.k_norm = up32(d, st_bf16(f, b + "self_attn.k_norm.weight", {128}));
        if (!d->flash) l.k_norm_ctx = up32(d, st_bf16(f, b + "self_attn.k_norm_ctx.weight", {128}));
        l.wq = backbone(d, st_bf16(f, b + "self_attn.q_proj.weight", {H * 128, C}), C);
        l.wk = backbone(d, st_bf16(f, b + "self_attn.k_proj.weight", {KV * 128, C}), C);
        l.wv = backbone(d, st_bf16(f, b + "self_attn.v_proj.weight", {KV * 128, C}), C);
        l.wo = backbone(d, st_bf16(f, b + "self_attn.o_proj.weight", {C, H * 128}), H * 128);
        if (d->flash) {
            auto load_conv = [&](const std::string& name) {
                return DLayer::Conv{
                    projection(d, st_bf16(f, b + name + ".kernel_projection.weight", {df2::dyn_width, C}), C),
                    up16(d, st_bf16(f, b + name + ".base_kernel", {2, 2, C}))};
            };
            l.attention_conv = load_conv("attention_conv");
            l.mlp_conv = load_conv("mlp_conv");
        } else {
            l.wkc = up16(d, st_bf16(f, b + "self_attn.k_proj_ctx.weight", {KV * 128, C}));
            l.wvc = up16(d, st_bf16(f, b + "self_attn.v_proj_ctx.weight", {KV * 128, C}));
        }
        {
            auto g = st_bf16(f, b + "mlp.gate_proj.weight", {F, C});
            const auto u = st_bf16(f, b + "mlp.up_proj.weight", {F, C});
            g.insert(g.end(), u.begin(), u.end());   // gate's rows, then up's
            l.wgu = backbone(d, g, C);
        }
        l.wd = backbone(d, st_bf16(f, b + "mlp.down_proj.weight", {C, F}), F);
        if (l.local) {   // (a window of 1024: a ring of 2048 rows, F32)
            l.cap = std::min(ctx, 2048);
            l.kc = dalloc<float>((size_t) slots * l.cap * KV * 128);
            l.vc = dalloc<float>((size_t) slots * l.cap * KV * 128);
            d->vram += (size_t) slots * l.cap * KV * 128 * 8;
        } else {         // (the whole context, or IK2_DRAFT_CTX's last positions: FP16)
            l.cap = d->cap;
            l.kc16 = dalloc<uint16_t>((size_t) slots * l.cap * KV * 128);
            l.vc16 = dalloc<uint16_t>((size_t) slots * l.cap * KV * 128);
            d->vram += (size_t) slots * l.cap * KV * 128 * 4;
        }
    }
    d->norm = up32(d, st_bf16(f, "norm.weight", {C}));
    if (d->flash) {
        auto& flash = *d->flash;
        flash.draws = dalloc<Draws>(slots);
        flash.mass = dalloc<DraftMass>(slots * (kBlock - 1));
        cu(cudaHostAlloc(&flash.host_draws, slots*sizeof(Draws), cudaHostAllocDefault), "draft draws");
        cu(cudaHostAlloc(&flash.host_mass, slots*(kBlock-1)*sizeof(DraftMass), cudaHostAllocDefault), "draft probabilities");
        flash.fc = projection(d, st_bf16(f, "fc.weight", {C, 4 * C}), 4 * C);
        flash.hidden_norm = up32(d, st_bf16(f, "hidden_norm.weight", {C}));
        flash.selector = projection(d, st_bf16(f, "candidate_selector.hidden_projection.weight", {R, C}), C);
        flash.pred = up16(d, st_bf16(f, "candidate_selector.predecessor_codebook", {df2::codebook_rows, R}));
        flash.succ = up16(d, st_bf16(f, "candidate_selector.successor_codebook", {df2::codebook_rows, R}));
        flash.packed = dalloc<uint16_t>(kMixRows * 4 * C);
        flash.dynamic = dalloc<float>(slots * kBlock * df2::dyn_width);
        flash.parts = dalloc<int>(slots * (kBlock - 1) * df2::parts * df2::top);
        flash.ids = dalloc<int>(slots * (kBlock - 1) * df2::top);
        d->mask = dalloc<float>(C);
        d->proj = dalloc<float>(slots * (kBlock - 1) * R);
        d->base = dalloc<float>(slots * (kBlock - 1) * df2::vocab);
    } else {
        d->mask = up32(d, st_bf16(f, "mask_embed", {C}));
        d->mk_dflare = up32(d, st_bf16(f, "markov_head.dflare_norm.weight", {C}));
        d->w1 = up16(d, st_bf16(f, "markov_head.markov_w1.weight", {d->V, R}));
        d->rnn_w = up16(d, st_bf16(f, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * C}));
        d->rnn_b = up32(d, st_bf16(f, "markov_head.rnn_proj.bias", {3 * R}));
        d->hout_w = st_bf16(f, "markov_head.hidden_out.weight", {C, R});
        d->hout_b = st_bf16(f, "markov_head.hidden_out.bias", {C});
    }
    // work buffers
    const size_t T = kBlock * (d->flash ? slots : 1);
    d->x = dalloc<float>(T * C), d->res = dalloc<float>(T * C), d->h = dalloc<float>(T * C);
    d->q = dalloc<float>(T * H * 128), d->o = dalloc<float>(T * H * 128);
    d->kk = dalloc<float>((d->flash ? T : kMixRows) * KV * 128), d->vv = dalloc<float>(T * KV * 128);
    d->gu = dalloc<float>(T * 2 * F);
    d->hb = dalloc<uint16_t>(std::max<size_t>(T * std::max<int64_t>(C, H * 128), (size_t) kMixRows * C));
    d->hb2 = dalloc<uint16_t>(T * F);
    // (online-softmax draft_attn's: only the chunks' outputs [64][H][8][128] and (max, sum))
    if (!d->flash) d->ctx_tmp = dalloc<float>((size_t) 2 * kMixRows * KV * 128);
    d->scratch = dalloc<float>((size_t) (d->flash ? slots * 8 : 64) * H * 8 * 130);
    d->mix = dalloc<float>((size_t) kMixRows * C);
    if (!d->flash) {
        d->f_rnn = dalloc<float>((size_t)slots * C);
        d->state = dalloc<float>(R), d->latent = dalloc<float>(R), d->proj = dalloc<float>(3 * R);
        d->rnn_in = dalloc<uint16_t>(2 * R + 2 * C);
    }
    d->x16 = dalloc<__half>(T * C);
    d->pos = dalloc<int>(T), d->ctx_pos = dalloc<int>(kMixRows), d->toks = dalloc<int>(T);
    cu(cudaHostAlloc(&d->h_out, T * sizeof(int), cudaHostAllocDefault), "h_out");
#ifndef STRATA_USE_HIP
    // Captured cuBLAS GEMMs share one workspace across all widths and depth buckets on d->s.
    if (!getenv("IK2_DRAFT_GRAPH") || atoi(getenv("IK2_DRAFT_GRAPH"))) {
        constexpr size_t workspace_bytes = 4 * 1024 * 1024;
        cb(cublasCreate(&d->blas), "proposal handle");
        cb(cublasSetStream(d->blas, d->s), "proposal stream");
        cb(cublasSetWorkspace(d->blas, dalloc<uint8_t>(workspace_bytes), workspace_bytes), "proposal workspace");
        d->vram += workspace_bytes;
    }
#endif
    d->d_meta = dalloc<int>(3 * slots);
    cu(cudaHostAlloc(&d->h_meta, 3 * slots * sizeof(int), cudaHostAllocDefault), "h_meta");
    return d;
}

void select_slot(Drafter* d, int slot) {
    const ptrdiff_t delta = slot - d->slot;
    if (!delta) return;
    for (auto& l : d->L) {
        const ptrdiff_t offset = delta * l.cap * d->KV * 128;
        if (l.kc16) {
            l.kc16 += offset;
            l.vc16 += offset;
        } else {
            l.kc += offset;
            l.vc += offset;
        }
    }
    if (!d->flash) d->f_rnn += delta * d->C;
    d->slot = slot;
}

bool probabilistic(const Drafter* d) { return d && d->flash && d->flash->sample; }

float* dflash2_mask(Drafter* d) { return d->flash ? d->mask : nullptr; }

void fold(Drafter* d, const uint16_t* lm_f16, int n_rows) {
    const int C = d->C, R = d->R;
    d->n_rows = n_rows;
    if (d->flash) {
        d->flash->head = lm_f16;
        return;
    }
    if (getenv("IK2_DRAFT_SHORTLIST") && atoi(getenv("IK2_DRAFT_SHORTLIST"))) {
        const int stride = (n_rows + 31) / 32;
        d->shortlist = dalloc<int>((kBlock - 1) * stride);
        d->shortlist_scores = dalloc<float>(stride);
    }
    // fold_w [R][n_rows] = (lm . hidden_out)^T, fold_b [n_rows] = lm . b (stored as FP16 to halve Markov traffic).
    // Compute chunked in F32 (like inkling2_mtp.py's 4096-row double() fold) so BF16 hout_w/b never underflow FP16
    // before the C=4608 inner product, then round once to FP16.
    std::vector<float> ht((size_t) R * C), hb(C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) ht[(size_t) r * C + c] = bf(d->hout_w[(size_t) c * R + r]);
    for (int c = 0; c < C; ++c) hb[c] = bf(d->hout_b[c]);
    float *dht = dalloc<float>(ht.size()), *dhb = dalloc<float>(hb.size());
    cu(cudaMemcpy(dht, ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    cu(cudaMemcpy(dhb, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    const int chunk = std::min(kFoldChunk, std::max(1, n_rows));
    float* dlm = dalloc<float>((size_t) chunk * C);
    float* dfw = dalloc<float>((size_t) R * chunk);
    float* dfb = dalloc<float>(chunk);
    d->fold_w = dalloc<uint16_t>((size_t) R * n_rows);
    d->fold_b = dalloc<uint16_t>(n_rows);
    d->base = dalloc<float>((size_t) (kBlock - 1) * n_rows);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < n_rows; r0 += chunk) {
        const int m = std::min(chunk, n_rows - r0);
        const size_t elems = (size_t) m * C;
        f16_to_f32_kernel<<<(unsigned) ((elems + 255) / 256), 256, 0, d->s>>>(lm_f16 + (size_t) r0 * C, elems, dlm);
        cb(cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, R, C, &one, dlm, C, dht, C, &zero, dfw, m), "fold w");
        cb(cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, C, &one, dlm, C, dhb, C, &zero, dfb, m), "fold b");   // (Sgemv as a 1-column Sgemm: hipblasSgemv faults on Windows)
        if (d->shortlist)
            fold_vocab_rows<<<(m * R + 255) / 256, 256, 0, d->s>>>(dfw, m, r0, R, d->fold_w);
        else
            pack_f16_strided_kernel<<<dim3((unsigned) ((m + 255) / 256), (unsigned) R), 256, 0, d->s>>>(
            dfw, m, n_rows, r0, d->fold_w);
        k::to_f16(dfb, (size_t) m, d->fold_b + r0, d->s);
    }
    cu(cudaStreamSynchronize(d->s), "fold");
    for (auto* p : {dht, dhb, dlm, dfw, dfb}) cu(cudaFree(p), "fold cleanup");
    d->vram += ((size_t) R + 1) * n_rows * 2 + (size_t) (kBlock - 1) * n_rows * 4;
}

#include "ik2_dflash2.cuh"

void write_context(Drafter* d, const float* taps, int ld, int r0, int m, int pos0) {
    if (d->flash) return df2::context(d, taps, ld, r0, m, pos0);
    const int C = d->C, KD = d->KV * 128;
    for (auto& l : d->L) {
        const int li = (int) (&l - d->L.data());
        for (int a = 0; a < m;) {
            const int at = (pos0 + a) % l.cap;
            const int n = std::min({kMixRows, m - a, l.cap - at});   // (a ring: a chunk ends where it wraps)
            iota_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, d->s>>>(d->ctx_pos, pos0 + a, n);
            k::mix_rms(taps + (size_t) (r0 + a) * C, d->taps, (size_t) ld * C, n, C, d->alpha + li * d->taps, l.dflare,
                       d->eps, d->mix, d->s);
            k::to_bf16(d->mix, (size_t) n * C, d->hb, d->s);
            float* kd = l.kc16 ? d->ctx_tmp : l.kc + (size_t) at * KD;
            float* vd = l.kc16 ? d->ctx_tmp + (size_t) kMixRows * KD : l.vc + (size_t) at * KD;
            gemm_bf(d, l.wkc, KD, C, d->hb, n, kd, KD);
            gemm_bf(d, l.wvc, KD, C, d->hb, n, vd, KD);
            k::rms(kd, l.k_norm_ctx, n * d->KV, 128, d->eps, kd, d->s);
            k::rope_neox(kd, n, d->KV, d->ctx_pos, d->theta, d->s);
            if (l.kc16) {
                k::to_f16(kd, (size_t) n * KD, l.kc16 + (size_t) at * KD, d->s);
                k::to_f16(vd, (size_t) n * KD, l.vc16 + (size_t) at * KD, d->s);
            }
            a += n;
        }
    }
}

void prefill_context(Drafter* d, const float* taps, int ld, int n, int pos, int end) {
    // Preserve the GEMM shape of every retained chunk, including the boundary chunk.
    if (d->flash && pos + n <= end - df2::window) return;
    write_context(d, taps, ld, 0, n, pos);
}

void commit_windows(Drafter* d, const float* taps, int ld, const std::vector<ContextWindow>& windows) {
    if (windows.empty()) return;
    if (windows.size() == 1 || !d->flash) {
        for (const auto& w : windows) {
            select_slot(d, w.slot);
            write_context(d, taps, ld, w.first, w.count, w.position);
            set_f_rnn(d, taps, ld, w.first + w.count - 1);
        }
        return;
    }

    select_slot(d, 0);
    df2::ContextRows rows{};
    int n = 0;
    for (const auto& w : windows) {
        for (int j = 0; j < w.count; ++j, ++n) {
            rows.tap[n] = w.first + j;
            rows.position[n] = w.position + j;
            rows.slot[n] = w.slot;
        }
    }
    auto& f = *d->flash;
    df2::gather_context<<<dim3(d->C / 256, n, 4), 256, 0, d->s>>>(taps, ld, rows, f.packed, d->ctx_pos);
    gemm_bf(d, f.fc, d->C, 4 * d->C, f.packed, n, d->mix, d->C);
    df2::norm<<<n, 256, 0, d->s>>>(d->mix, nullptr, f.hidden_norm, d->C, d->mix);
    backbone_input(d, d->mix, n * d->C, d->hb);
    const int kd = d->KV * df2::head_dim;
    for (auto& l : d->L) {
        project_backbone(d, l.wk, kd, d->C, d->hb, n, d->kk, kd);
        project_backbone(d, l.wv, kd, d->C, d->hb, n, d->vv, kd);
        df2::prepare_keys(d, d->kk, l.k_norm, n, d->KV, d->ctx_pos);
        df2::scatter_context<<<dim3(kd / 256, n), 256, 0, d->s>>>(d->kk, d->vv, rows, l.cap, kd, l.kc, l.vc);
    }
}

void set_f_rnn(Drafter* d, const float* taps, int ld, int row) {
    if (d->flash) return;
    k::mix_rms(taps + (size_t) row * d->C, d->taps, (size_t) ld * d->C, 1, d->C, d->alpha + d->L.size() * d->taps, d->mk_dflare,
               d->eps, d->f_rnn, d->s);
}

static void dspark_block(Drafter* d, int proposals, int P, const uint16_t* lm_f16, int n_rows, int n_vocab, int splits) {
    const int C = d->C, T = kBlock;
    const size_t TC = (size_t) T * C;
    const int shortlist_n = (n_vocab + 31) / 32;
    // pre-norm layers, the residual carried apart (res = x + res before each norm)
    for (size_t i = 0; i < d->L.size(); ++i) {
        DLayer& l = d->L[i];
        if (i > 0) k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.in_norm, T, C, d->eps, d->h, d->s);
        layer_attn(d, l, P, d->x, splits);
        k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.post_norm, T, C, d->eps, d->h, d->s);
        layer_mlp(d, l, d->x);
    }
    k::add(d->x, d->res, TC, d->res, d->s);
    k::rms(d->res, d->norm, T, C, d->eps, d->h, d->s);   // the draft hidden, rows 1 .. 7 used
    // base logits of rows 1 .. 7 through the target's head
    k::to_f16(d->h + C, (size_t) proposals * C, (uint16_t*) d->x16, d->s);
    if (d->head_q8)
        k::q8_tile(d->head_q8, d->head_q8d, n_rows, C, (const uint16_t*) d->x16, proposals, d->base, n_rows, d->s);
    else
        gemm_h(d, lm_f16, n_rows, C, d->x16, proposals, d->base, n_rows);
    if (d->shortlist)
        draft_shortlist<<<dim3((n_vocab + 127) / 128, proposals), 128, 0, d->s>>>(
            d->base, n_rows, n_vocab, shortlist_n, d->shortlist);
    // the Markov head, a token at a time (note: k::rnn_gates keeps state/latent in F32 whereas SGLang mtp.py
    // rounds intermediate gate/tanh ops to BF16; fused argmax_gemv_f16 reads FP16 fold_w/fold_b + F32 base/latent)
    cu(cudaMemsetAsync(d->state, 0, d->R * 4, d->s), "state");
    for (int t = 0; t < proposals; ++t) {
        k::rnn_inputs(d->state, d->w1, d->toks + t, d->h + (size_t) (t + 1) * C, d->f_rnn, d->R, C, d->rnn_in, d->s);
        gemm_bf(d, d->rnn_w, 3 * d->R, 2 * d->R + 2 * C, d->rnn_in, 1, d->proj, 3 * d->R);
        k::rnn_gates(d->proj, d->rnn_b, d->R, d->state, d->latent, d->s);
        if (d->shortlist) {
            const int* ids = d->shortlist + t * shortlist_n;
            shortlist_gemv<<<(shortlist_n + 3) / 4, 128, 0, d->s>>>(ids, shortlist_n, d->fold_w, d->latent,
                d->base + (size_t)t * n_rows, d->fold_b, d->R, d->shortlist_scores);
            k::argmax_sum(d->shortlist_scores, nullptr, nullptr, shortlist_n, d->toks + t + 1, d->s);
            shortlist_pick<<<1, 1, 0, d->s>>>(d->toks + t + 1, ids);
        } else {
            k::argmax_gemv_f16(d->fold_w, n_rows, d->latent, d->base + (size_t) t * n_rows, d->fold_b, d->R, n_vocab,
                           d->toks + t + 1, d->s);
        }
    }
}

void propose(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows, int n_vocab,
             int* out, int proposals, const Draws* draws, DraftMass* mass) {
    if (draws) *d->flash->host_draws = *draws;
    const int C = d->C, T = kBlock;
    if (d->flash && (n_vocab != df2::vocab || n_rows < n_vocab || anchor < 0 || anchor >= n_vocab))
        throw std::runtime_error("DFlash2 requires the Inkling-2-Mini vocabulary");
    static const bool use_graph = !getenv("IK2_DRAFT_GRAPH") || atoi(getenv("IK2_DRAFT_GRAPH"));
    bool graph_ok = use_graph && (d->head_q8 || d->flash) && proposals >= 1 && proposals < kBlock;
#ifdef STRATA_USE_HIP
    static const bool bf16_mv = !getenv("IK2_DRAFT_BF16_MV") || atoi(getenv("IK2_DRAFT_BF16_MV"));
    graph_ok = graph_ok && bf16_mv && C % 128 == 0 && d->F % 128 == 0 && (2 * d->R + 2 * C) % 128 == 0;
#endif
    Drafter::PropGraph* capture = nullptr;
    int splits = 0;
    if (graph_ok) {
        const int bkt = (d->flash || P < 256) ? 0 : (P < 1024 ? 1 : (P < 4096 ? 2 : 3));
        static constexpr int kSplits[4] = {2, 8, 16, 64};
        d->h_meta[0] = P;
        d->h_meta[1] = anchor;
        auto& G = d->graphs[d->slot].width[proposals][draws ? 4 : bkt];
        if (G.exec && (G.anchor_h != anchor_h || G.n_rows != n_rows || G.n_vocab != n_vocab)) {
            cu(cudaGraphExecDestroy(G.exec), "pgraph destroy exec");
            cu(cudaGraphDestroy(G.graph), "pgraph destroy");
            G = {};
        }
        if (G.exec) {
            cu(cudaGraphLaunch(G.exec, d->s), "pgraph launch");
            cu(cudaStreamSynchronize(d->s), "propose");
            if (mass) std::copy_n(d->flash->host_mass, proposals, mass);
            std::memcpy(out, d->h_out, proposals * sizeof(int));
            return;
        }
        capture = &G;
        splits = kSplits[bkt];
        G.anchor_h = anchor_h;
        G.n_rows = n_rows;
        G.n_vocab = n_vocab;
        cu(cudaStreamBeginCapture(d->s, cudaStreamCaptureModeGlobal), "pgraph begin");
        cu(cudaMemcpyAsync(d->d_meta, d->h_meta, 2 * sizeof(int), cudaMemcpyHostToDevice, d->s), "meta h2d");
        init_block_kernel_d<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, d->s>>>(
            anchor_h, d->mask, d->res, d->pos, d->toks, T, C, d->d_meta);
    } else {
        // Initialize the residual stream [anchor_h; mask x 7], block positions, and anchor token in one kernel.
        // Layer 0 reads d->res and overwrites d->x, so d->x does not need pre-filling.
        init_block_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, d->s>>>(
            anchor_h, d->mask, d->res, d->pos, d->toks, T, C, P, anchor);
    }
    if (draws) {
        auto& f = *d->flash;
        cu(cudaMemcpyAsync(f.draws, f.host_draws, sizeof(Draws), cudaMemcpyHostToDevice, d->s), "draws");
        df2::block(d, proposals, 1, nullptr, true);
        cu(cudaMemcpyAsync(f.host_mass, f.mass, proposals*sizeof(DraftMass), cudaMemcpyDeviceToHost, d->s), "mass");
    } else if (d->flash) df2::block(d, proposals);
    else dspark_block(d, proposals, P, lm_f16, n_rows, n_vocab, splits);
    cu(cudaMemcpyAsync(d->h_out, d->toks + 1, proposals * 4, cudaMemcpyDeviceToHost, d->s), "tokens");
    if (capture) {
        cu(cudaStreamEndCapture(d->s, &capture->graph), "pgraph end");
        cu(cudaGraphInstantiate(&capture->exec, capture->graph, 0), "pgraph instantiate");
        cu(cudaGraphLaunch(capture->exec, d->s), "pgraph launch");
    }
    cu(cudaStreamSynchronize(d->s), "propose");
    if (mass) std::copy_n(d->flash->host_mass, proposals, mass);
    std::memcpy(out, d->h_out, proposals * sizeof(int));
}

void propose_batch(Drafter* d, int active, const int* slots, const int* positions, const float* embeddings,
                   const uint16_t* lm_f16, int n_rows, int n_vocab, int window, int* tokens,
                   const Draws* draws, DraftMass* mass) {
    if (!d->flash || active == 1) {
        for (int i = 0; i < active; ++i) {
            select_slot(d, slots[i]);
            int* block = tokens + i * window;
            propose(d, block[0], embeddings + i * d->C, positions[i], lm_f16, n_rows, n_vocab, block + 1, window - 1,
                    draws ? draws+i : nullptr, mass ? mass+i*(window-1) : nullptr);
        }
        return;
    }
    select_slot(d, 0);
    for (int i = 0; i < active; ++i) {
        d->h_meta[2*i] = positions[i];
        d->h_meta[2*i+1] = tokens[i * window];
        d->h_meta[2*active+i] = slots[i];
    }
    const size_t bytes = active * window * sizeof(int);
    if (draws) std::copy_n(draws, active, d->flash->host_draws);
    auto launch = [&] {
        if (draws) cu(cudaMemcpyAsync(d->flash->draws, d->flash->host_draws, active*sizeof(Draws), cudaMemcpyHostToDevice, d->s), "batch draws");
        cu(cudaMemcpyAsync(d->d_meta, d->h_meta, 3 * active * sizeof(int), cudaMemcpyHostToDevice, d->s), "batch metadata");
        init_block_kernel_d<<<dim3((d->C + 255) / 256, kBlock, active), 256, 0, d->s>>>(
            embeddings, d->mask, d->res, d->pos, d->toks, kBlock, d->C, d->d_meta, window);
        df2::block(d, window - 1, active, d->d_meta + 2*active, draws != nullptr);
        if (draws) cu(cudaMemcpyAsync(d->flash->host_mass, d->flash->mass, active*(window-1)*sizeof(DraftMass),
                                     cudaMemcpyDeviceToHost, d->s), "batch probabilities");
        cu(cudaMemcpyAsync(d->h_out, d->toks, bytes, cudaMemcpyDeviceToHost, d->s), "batch tokens");
    };
    static const bool use_graph = !getenv("IK2_DRAFT_GRAPH") || atoi(getenv("IK2_DRAFT_GRAPH"));
    if (use_graph) {
        auto& g = d->batch_graph[active][window][draws != nullptr];
        if (g.exec && g.anchor_h != embeddings) {
            cu(cudaGraphExecDestroy(g.exec), "batch graph destroy exec");
            cu(cudaGraphDestroy(g.graph), "batch graph destroy");
            g = {};
        }
        if (!g.exec) {
            cu(cudaStreamBeginCapture(d->s, cudaStreamCaptureModeGlobal), "batch graph begin");
            launch();
            cu(cudaStreamEndCapture(d->s, &g.graph), "batch graph end");
            cu(cudaGraphInstantiate(&g.exec, g.graph, 0), "batch graph instantiate");
            g.anchor_h = embeddings;
        }
        cu(cudaGraphLaunch(g.exec, d->s), "batch graph launch");
    } else launch();
    cu(cudaStreamSynchronize(d->s), "propose_batch");
    if (mass) std::copy_n(d->flash->host_mass, active*(window-1), mass);
    std::memcpy(tokens, d->h_out, bytes);
}

std::vector<CacheSpan> cache_spans(const Drafter* d) {
    std::vector<CacheSpan> spans;
    if (!d->flash) spans.push_back({d->f_rnn, 1, d->C * sizeof(float)});
    for (const auto& l : d->L) {
        const size_t stride = d->KV * 128 * (l.kc16 ? sizeof(uint16_t) : sizeof(float));
        spans.push_back({l.kc16 ? (void*)l.kc16 : (void*)l.kc, (size_t)l.cap, stride});
        spans.push_back({l.vc16 ? (void*)l.vc16 : (void*)l.vc, (size_t)l.cap, stride});
    }
    return spans;
}

size_t bytes(const Drafter* d) { return d->vram; }
void set_head_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d) { d->head_q8 = q8, d->head_q8d = q8d; }

}  // namespace ik2::draft
