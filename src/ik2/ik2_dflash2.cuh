// Included by ik2_draft.cu to share the BF16 GEMM, proposal graph and cache storage.
namespace df2 {
namespace {
constexpr int heads = 32, kv_heads = 8, head_dim = 128, qdim = heads * head_dim, kdim = kv_heads * head_dim;
constexpr int ff = 6912, window = 2048;

void round_values(Drafter* d, float* x, int count) {
    round_array<<<(count + 255) / 256, 256, 0, d->s>>>(x, count);
}

void prepare_keys(Drafter* d, float* x, const float* weight, int rows, int nh, const int* pos) {
    norm<<<rows * nh, 256, 0, d->s>>>(x, nullptr, weight, head_dim, x);
    k::rope_neox(x, rows, nh, pos, d->theta, d->s);
    round_values(d, x, rows * nh * head_dim);
}

void context(Drafter* d, const float* taps, int ld, int first, int rows, int position) {
    auto& f = *d->flash;
    const int cap = d->L.front().cap;
    while (rows) {
        const int at = position % cap;
        const int n = std::min({rows, kMixRows, cap - at});
        gather<<<dim3(width / 256, n, 4), 256, 0, d->s>>>(taps, ld, first, f.packed);
        gemm_bf(d, f.fc, width, 4 * width, f.packed, n, d->mix, width);
        norm<<<n, 256, 0, d->s>>>(d->mix, nullptr, f.hidden_norm, width, d->mix);
        backbone_input(d, d->mix, n * width, d->hb);
        iota_kernel<<<(n + 255) / 256, 256, 0, d->s>>>(d->ctx_pos, position, n);
        for (auto& l : d->L) {
            float* keys = l.kc + size_t(at) * kdim;
            float* values = l.vc + size_t(at) * kdim;
            project_backbone(d, l.wk, kdim, width, d->hb, n, keys, kdim);
            project_backbone(d, l.wv, kdim, width, d->hb, n, values, kdim);
            prepare_keys(d, keys, l.k_norm, n, kv_heads, d->ctx_pos);
            round_values(d, values, n * kdim);
        }
        first += n;
        position += n;
        rows -= n;
    }
}

void conv_input(Drafter* d, DLayer::Conv weights, int T) {
    k::to_bf16(d->h, T * width, d->hb, d->s);
    gemm_bf(d, weights.projection, dyn_width, width, d->hb, T, d->flash->dynamic, dyn_width);
    conv<<<dim3(width / 256, T), 256, 0, d->s>>>(d->h, d->flash->dynamic, weights.base, 0, d->x);
    backbone_input(d, d->x, T * width, d->hb);
}

void conv_output(Drafter* d, DLayer::Conv weights, int T) {
    conv<<<dim3(width / 256, T), 256, 0, d->s>>>(d->h, d->flash->dynamic, weights.base, 1, d->x);
}

void block(Drafter* d, int proposals, int active = 1, const int* slots = nullptr, bool sample = false) {
    auto& f = *d->flash;
    const int T = active * kBlock;
    const int rows = active * proposals;
    for (auto& l : d->L) {
        if (&l == &d->L.front())
            norm<<<T, 256, 0, d->s>>>(d->res, nullptr, l.in_norm, width, d->h);
        else
            norm<<<T, 256, 0, d->s>>>(d->x, d->res, l.in_norm, width, d->h);
        conv_input(d, l.attention_conv, T);
        project_backbone(d, l.wq, qdim, width, d->hb, T, d->q, qdim);
        project_backbone(d, l.wk, kdim, width, d->hb, T, d->kk, kdim);
        project_backbone(d, l.wv, kdim, width, d->hb, T, d->vv, kdim);
        prepare_keys(d, d->q, l.q_norm, T, heads, d->pos);
        prepare_keys(d, d->kk, l.k_norm, T, kv_heads, d->pos);
        round_values(d, d->vv, T * kdim);
        k::dflash2_attn(d->q, l.kc, l.vc, d->pos, window - 1, d->kk, d->vv, kBlock, heads, kv_heads,
                        1.f / std::sqrt(float(head_dim)), 8, d->scratch, d->o, d->s, l.cap, active, slots);
        backbone_input(d, d->o, T * qdim, d->hb);
        project_backbone(d, l.wo, width, qdim, d->hb, T, d->h, width);
        conv_output(d, l.attention_conv, T);
        norm<<<T, 256, 0, d->s>>>(d->x, d->res, l.post_norm, width, d->h);
        conv_input(d, l.mlp_conv, T);
        project_backbone(d, l.wgu, 2 * ff, width, d->hb, T, d->gu, 2 * ff);
        round_values(d, d->gu, T * 2 * ff);
        k::silu_mul_bf16(d->gu, T, ff, d->hb2, d->s);
        if (d->bits != 16) bf16_to_half<<<(T*ff+255)/256, 256, 0, d->s>>>(d->hb2, T*ff);
        project_backbone(d, l.wd, width, ff, d->hb2, T, d->h, width);
        conv_output(d, l.mlp_conv, T);
    }
    norm<<<T, 256, 0, d->s>>>(d->x, d->res, d->norm, width, d->h);
    float* hidden = d->h + width;
    if (active > 1) {
        cu(cudaMemcpy2DAsync(d->x, proposals * width * sizeof(float), hidden, kBlock * width * sizeof(float),
                             proposals * width * sizeof(float), active, cudaMemcpyDeviceToDevice, d->s), "selector rows");
        hidden = d->x;
    }
    k::to_bf16(hidden, rows * width, d->hb, d->s);
    gemm_bf(d, f.selector, rank, width, d->hb, rows, d->proj, rank);
    k::to_f16(hidden, rows * width, (uint16_t*)d->x16, d->s);
    if (d->head_q8) {
#ifdef STRATA_USE_HIP
        // Use the same WMMA reduction for the head at every active width.
        k::rdna3_planar_gemm(d->head_q8, d->head_q8d, vocab, width, (const uint16_t*)d->x16, rows, d->base, vocab, d->s);
#else
        k::q8_tile(d->head_q8, d->head_q8d, vocab, width, (const uint16_t*)d->x16, rows, d->base, vocab, d->s);
#endif
    } else gemm_h(d, f.head, vocab, width, d->x16, rows, d->base, vocab);
    radix_parts<<<dim3(parts, rows), 256, 0, d->s>>>(d->base, vocab, f.parts);
    merge_parts<<<rows, 128, 0, d->s>>>(d->base, vocab, f.parts, f.ids);
    if (sample)
        walk<true><<<active, top * 32, 0, d->s>>>(d->base, vocab, f.ids, d->proj, f.pred, f.succ, proposals,
                                                  d->toks, f.draws, f.mass);
    else
        walk<<<active, top * 32, 0, d->s>>>(d->base, vocab, f.ids, d->proj, f.pred, f.succ, proposals, d->toks);
}
} // namespace
} // namespace df2
