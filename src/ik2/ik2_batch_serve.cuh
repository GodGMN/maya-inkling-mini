// One line from the Maya frontend owns one slot until DONE. Only the scheduler touches GPU state.
struct BatchRequest {
    int rid = 0, limit = 0, generated = 0, pos = 0, token = 0;
    std::vector<int> ids, stops;
    std::string image_path;
    SampleCfg sampling;
    std::mt19937_64 rng;
    double start = 0, decode_start = 0;
    bool ready = false, seeded = false;
    int draft_windows = 0, verified_tokens = 0;

    int pick(const float* logits, int vocab) {
        if (sampling.temp <= 0) return std::max_element(logits, logits + vocab) - logits;
        std::vector<std::pair<float, int>> candidates(vocab);
        for (int i = 0; i < vocab; ++i) candidates[i] = {logits[i], i};
        return sample_from(sample_dist(candidates, sampling), rng);
    }
};

struct BatchInbox {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> lines;
    bool eof = false;
};

void serve_sequences(Model& M, int width, int ctx, int chunk, int image_token, std::vector<int> eos,
                     ik2::draft::Drafter* drafter, const uint16_t* draft_head) {
    const Config& c = M.c;
    Work& W = dev_work(0);
    if (!g_fast || g_managed_experts || host_plan.forced || !M.out_w.q8 ||
        !std::all_of(M.L.begin(), M.L.end(), [](const Layer& l) {
            return l.wo.q8 && (l.sliding ? l.wqkvrg.q8 != nullptr : l.wqrg.q8 && l.wk.q8 && l.wv.q8);
        })) die("IK2_BATCH requires resident experts and planar Q8 projections");
    auto base = M.V;
    const int KD = c.KV * c.D;
    auto bind_slot = [&](int slot, bool clear) {
        for (int i = 0; i < c.n_visit; ++i) {
            Visit& v = M.V[i] = sequence_visit(base[i], M.L[base[i].l], c, slot);
            if (clear) {
                for (float* h : {v.h_k, v.h_v}) CU(cudaMemsetAsync(h, 0, 3 * KD * sizeof(float), g_s));
                for (float* h : {v.h_attn, v.h_mlp}) CU(cudaMemsetAsync(h, 0, 3 * c.C * sizeof(float), g_s));
            }
        }
    };
    const int draft_width = drafter ? std::clamp(getenv("IK2_DRAFT_WIDTH") ? atoi(getenv("IK2_DRAFT_WIDTH")) : 7,
                                                  2, kBlockMax) : 1;
    const int max_rows = width * draft_width;
    k::ConvRec* rollback = nullptr;
    if (drafter) {
        const int channels[] = {KD, KD, c.C, c.C};
        for (auto& v : base) {
            for (int i = 0; i < 4; ++i) {
                v.rec_old[i] = dalloc<float>(size_t(width) * 3 * channels[i]);
                v.rec_x[i] = dalloc<float>(size_t(width) * kBlockMax * channels[i]);
            }
        }
        std::vector<k::ConvRec> records;
        for (int slot = 0; slot < width; ++slot) {
            for (const Visit& visit : base) {
                const Visit v = sequence_visit(visit, M.L[visit.l], c, slot);
                float* histories[] = {v.h_k, v.h_v, v.h_attn, v.h_mlp};
                for (int i = 0; i < 4; ++i)
                    records.push_back({v.rec_old[i], v.rec_x[i], histories[i], channels[i]});
            }
        }
        rollback = dalloc<k::ConvRec>(records.size());
        CU(cudaMemcpy(rollback, records.data(), records.size() * sizeof(k::ConvRec), cudaMemcpyHostToDevice));
    }
    BlockBufs buf;
    buf.emb_h = W.emb_h; buf.pos_h = W.pos_h; buf.lp_msg = W.lp_msg_all;
    buf.blob = W.blob; buf.tok = W.tok; buf.inv = W.inv;
    buf.sg = W.sg; buf.su = W.su; buf.wt = W.wt;
#ifndef STRATA_USE_HIP
    buf.row = W.batch_group_row;
    if (buf.row) buf.n = W.gidx;
#endif
    CU(cudaHostAlloc(&buf.logits_h, (size_t)max_rows * c.n_vocab * sizeof(float), cudaHostAllocPortable));
    const size_t candidate_ints = (size_t)max_rows * (2 + 2 * kCandCap);
    buf.cand_d = dalloc<int>(candidate_ints);
    CU(cudaHostAlloc(&buf.cand_h, candidate_ints * sizeof(int), cudaHostAllocPortable));
    const bool use_candidates = !getenv("IK2_CAND") || std::atoi(getenv("IK2_CAND"));
    SequenceRows rows;
#ifndef STRATA_USE_HIP
    rows.native_fp4 = drafter && k::use_blackwell_nvfp4();
#endif
    rows.device_slots = dalloc<int>(max_rows);
    std::vector<int> tokens(max_rows);
    struct VerifyGraph {
        FastGraph graph;
        std::vector<int> slots, buckets;
        OutMode mode = OutMode::kNone;
    };
    VerifyGraph verify_graphs[kBlockMax + 1][3];  // lone-slot width and attention depth bucket
    auto inbox = std::make_shared<BatchInbox>();
    std::thread([inbox] {
        std::string line;
        while (std::getline(std::cin, line)) {
            const bool quit = line == "QUIT";
            std::lock_guard<std::mutex> lock(inbox->mutex);
            inbox->lines.push_back(std::move(line));
            inbox->changed.notify_one();
            if (quit) break;
        }
        std::lock_guard<std::mutex> lock(inbox->mutex);
        inbox->eof = true;
        inbox->changed.notify_one();
    }).detach();
    std::vector<std::unique_ptr<BatchRequest>> slots(width);
    std::deque<std::unique_ptr<BatchRequest>> waiting;
    std::vector<SpecCtl> solo_ctl(width);
    for (auto& ctl : solo_ctl) {
        if (getenv("IK2_DRAFT_WIDTH")) ctl.wmax = draft_width;
        ctl.adapt = !getenv("IK2_DRAFT_ADAPT") || atoi(getenv("IK2_DRAFT_ADAPT"));
    }
    const bool stats = getenv("IK2_STATS") != nullptr;
    auto done = [&](BatchRequest& r, const char* reason) {
        if (stats && r.draft_windows)
            std::fprintf(stderr, "draft request %d: %d tokens in %d windows\n", r.rid, r.verified_tokens, r.draft_windows);
        const double t = now(), start = r.ready ? r.decode_start : t;
        std::printf("DONE %d %d %zu %.1f %.1f %s 0 0 0\n", r.rid, r.generated, r.ids.size(),
                    (start - r.start) * 1e3, (t - start) * 1e3, reason);
    };
    auto finish = [&](int s, const char* reason) {
        // Keep this slot's measured kernel times. Acceptance and probe history belong to the request.
        auto& ctl = solo_ctl[s];
        ctl.S = ctl.F = 1;
        ctl.plain_run = 0;
        ctl.steps = ctl.plain = ctl.drafted = ctl.accepted = 0;
        std::fill_n(ctl.hist, kBlockMax, 0);
        done(*slots[s], reason);
        slots[s].reset();
    };
    auto emit = [&](int s, int token) {
        BatchRequest& r = *slots[s];
        r.token = token;
        ++r.generated;
        std::printf("T %d %d\n", r.rid, token);
        if (std::find(r.stops.begin(), r.stops.end(), token) != r.stops.end()) finish(s, "stop");
        else if (r.generated == r.limit) finish(s, "length");
    };
    std::mt19937_64 seeds(1234);
    auto admit = [&](std::istringstream& in, bool image) {
        auto request = std::make_unique<BatchRequest>();
        BatchRequest& r = *request;
        r.stops = eos;
        r.rng.seed(seeds());
        in >> r.limit;
        std::string word, ids;
        while (in >> word) {
            const size_t eq = word.find('=');
            if (eq == std::string::npos) {
                if (image && r.image_path.empty()) r.image_path = word;
                else ids = word;
                continue;
            }
            const std::string key = word.substr(0, eq);
            const char* value = word.c_str() + eq + 1;
            if (key == "rid") r.rid = std::atoi(value);
            else if (key == "temperature") r.sampling.temp = std::atof(value);
            else if (key == "top_p") r.sampling.top_p = std::atof(value);
            else if (key == "top_k") r.sampling.top_k = std::atoi(value);
            else if (key == "seed" && std::atoll(value) >= 0) { r.rng.seed(std::atoll(value)); r.seeded = true; }
            else if (key == "stop_token") r.stops.push_back(std::atoi(value));
        }
        bool valid = !ids.empty();
        for (const char* p = ids.c_str(); *p && valid;) {
            char* end = nullptr;
            const long token = std::strtol(p, &end, 10);
            valid = end != p && token >= 0 && token < c.n_vocab && (!*end || (*end == ',' && end[1]));
            if (valid) r.ids.push_back((int)token);
            p = *end ? end + 1 : end;
        }
        const auto free = std::find(slots.begin(), slots.end(), nullptr);
        if (!valid || !r.rid || r.limit < 0 || r.ids.size() + r.limit > (size_t)ctx || (!drafter && free == slots.end())) {
            std::printf("ERR %d %s\n", r.rid, free == slots.end() ? "all slots busy" : "invalid prompt or context length");
            return;
        }
        r.start = now();
        if (!waiting.empty() || free == slots.end()) waiting.push_back(std::move(request));
        else *free = std::move(request);
    };

    std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=1 cache_slots=%d\n",
                kv_format_name(), ctx, width);
    std::printf("READY %d stop multi=%d\n", ctx, width);
    std::fflush(stdout);
    int next_prefill = 0;
    double t_exp = 0;
    for (;;) {
        std::deque<std::string> lines;
        {
            std::unique_lock<std::mutex> lock(inbox->mutex);
            const bool idle = std::none_of(slots.begin(), slots.end(), [](const auto& r) { return bool(r); });
            if (idle && waiting.empty()) inbox->changed.wait(lock, [&] { return inbox->eof || !inbox->lines.empty(); });
            if (idle && waiting.empty() && inbox->eof && inbox->lines.empty()) break;
            lines.swap(inbox->lines);
        }
        for (const auto& line : lines) {
            std::istringstream in(line);
            std::string cmd;
            in >> cmd;
            if (cmd == "QUIT") return;
            if (cmd == "GEN" || cmd == "GENI") admit(in, cmd == "GENI");
            if (cmd == "STOP") {
                int rid;
                const bool tagged = bool(in >> rid);
                for (auto it = waiting.begin(); it != waiting.end();) {
                    if (!tagged || (*it)->rid == rid) {
                        done(**it, "cancel");
                        it = waiting.erase(it);
                    } else ++it;
                }
                for (int s = 0; s < width; ++s)
                    if (slots[s] && (!tagged || slots[s]->rid == rid)) finish(s, "cancel");
            }
        }
        for (auto& slot : slots) {
            if (waiting.empty()) break;
            if (!slot) {
                slot = std::move(waiting.front());
                waiting.pop_front();
            }
        }
        // Work on at most one prefill chunk between decode steps. Its picture rows live only during prefill.
        for (int i = 0; i < width; ++i) {
            const int s = (next_prefill + i) % width;
            if (!slots[s] || slots[s]->ready) continue;
            next_prefill = (s + 1) % width;
            BatchRequest& r = *slots[s];
            bind_slot(s, r.pos == 0);
            g_img.clear();
            if (!r.image_path.empty()) {
                std::vector<uint64_t> sig(r.ids.size(), 0);
                const std::string err = load_pictures(r.image_path, r.ids, image_token, c.C, g_img, sig);
                if (!err.empty()) {
                    std::printf("ERR %d %s\n", r.rid, err.c_str());
                    slots[s].reset();
                    break;
                }
            }
            const int n = std::min(chunk, (int)r.ids.size() - r.pos);
            const bool last = r.pos + n == (int)r.ids.size();
            std::vector<float> logits;
            // A one-token prefill must use this slot's addresses, including after a slot change.
            const bool graphs = g_graph;
            g_graph = false;
            forward(M, r.ids.data() + r.pos, n, r.pos, false, (last || (drafter && n == 1)) ? OutMode::kLogits : OutMode::kNone,
                    {}, logits, nullptr, t_exp);
            g_graph = graphs;
            if (drafter) {
                ik2::draft::select_slot(drafter, s);
                ik2::draft::prefill_context(drafter, g_taps, g_taps_ld, n, r.pos, r.ids.size());
                if (last) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
            }
            r.pos += n;
            const double ms = (now() - r.start) * 1e3;
            std::printf("PP %d %d %zu %.1f %.1f 0\n", r.rid, r.pos, r.ids.size(), ms, 1e3 * r.pos / ms);
            if (last) {
                r.decode_start = now();
                r.ready = true;
                if (r.limit == 0) finish(s, "length");
                else emit(s, r.pick(logits.data(), c.n_vocab));
            }
            break;
        }
        std::fflush(stdout);
        rows.slots.clear();
        rows.buckets.clear();
        OutMode mode = OutMode::kArgmax;
        // The output limit is applied by emit; only the KV capacity bounds speculative input.
        int B = draft_width;
        std::vector<int> active, positions;
        for (int s = 0; s < width; ++s) {
            if (!slots[s] || !slots[s]->ready) continue;
            active.push_back(s);
            positions.push_back(slots[s]->pos);
            B = std::min(B, ctx - slots[s]->pos);
        }
        if (active.empty()) continue;
        const bool solo = drafter && active.size() == 1;
        if (solo) {
            auto& r = *slots[active.front()];
            B = solo_ctl[active.front()].choose(std::min(ctx - r.pos, r.limit - r.generated));
        }
        const double step_start = solo ? now() : 0;
        rows.window = drafter ? B : 0;
        for (int s : active) {
            const auto& r = *slots[s];
            const size_t first = rows.slots.size();
            tokens[first] = r.token;
            for (int j = 0; j < B; ++j) {
                rows.slots.push_back(s);
                rows.buckets.push_back(decode_bucket(r.pos + j));
                W.pos_h[first + j] = r.pos + j;
            }
            if (r.sampling.temp > 0) mode = use_candidates ? OutMode::kCand : OutMode::kLogits;
        }
        M.V = base;
        const int T = rows.slots.size();
        const size_t rb = (size_t)c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        ik2::draft::DraftMass mass[kBlockMax * (kBlockMax-1)];
        const bool sample = B > 1 && mode != OutMode::kArgmax && ik2::draft::probabilistic(drafter);
        if (B > 1) {
            const int count = active.size();
            for (int i = 0; i < count; ++i)
                k::decode_host(M.embd_type, M.embd + size_t(tokens[i * B]) * rb, c.C, W.emb_h + i * c.C);
            CU(cudaMemcpyAsync(W.tmp, W.emb_h, size_t(count) * c.C * sizeof(float), cudaMemcpyHostToDevice, g_s));
            k::rms(W.tmp, M.embd_norm, count, c.C, c.eps, W.xq, g_s);
            ik2::draft::Draws draws[kBlockMax];
            if (sample) for (int i = 0; i < count; ++i) {
                auto& request = *slots[active[i]];
                if (request.sampling.temp <= 0) draws[i] = {};
                else if (request.seeded) {
                    // Preserve seeded token streams by peeking at the target uniforms.
                    auto rng = request.rng;
                    draws[i] = draft_draws(request.sampling.temp, B-1, rng);
                } else {
                    draws[i] = draft_draws(request.sampling.temp, B-1, request.rng);
                }
            }
            ik2::draft::propose_batch(drafter, count, active.data(), positions.data(), W.xq,
                                      draft_head, c.n_rows, c.n_vocab, B, tokens.data(), sample ? draws : nullptr, sample ? mass : nullptr);
        }
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t)tokens[t] * rb, c.C, W.emb_h + (size_t)t * c.C);
        const bool single = T == 1;
        const bool solo_block = solo && rows.native_fp4 && !single;
        const int end = positions.front() + B;
        const int block_bucket = solo_block ? (end < 256 ? 0 : end < 1024 ? 1 : 2) : -1;
        if (single || solo_block) bind_slot(rows.slots.front(), false);
        auto launch = [&] {
            k::enable_grouped_tensor_products((drafter && B > 1) || allow_short_batch_products());
            if (single) fast_token_launches(M, W, mode, rows.buckets.front());
            else if (solo_block) fast_block_launches(M, W, T, mode, buf, nullptr, block_bucket);
            else fast_block_launches(M, W, T, mode, buf, &rows);
        };
        CU(cudaMemcpyAsync(rows.device_slots, rows.slots.data(), T * sizeof(int), cudaMemcpyHostToDevice, g_s));
        if (g_abl_tokens >= 0 && g_abl_srow && T <= kAblRows) {   // each row's ablation strength by its generated tokens
            for (int t = 0; t < T; ++t) {
                const BatchRequest& r = *slots[rows.slots[t]];
                g_abl_srow_h[t] = W.pos_h[t] - (int) r.ids.size() < g_abl_tokens ? 1.0f : 0.0f;
            }
            CU(cudaMemcpyAsync(g_abl_srow, g_abl_srow_h, T * sizeof(float), cudaMemcpyHostToDevice, g_s));
        }
        if (g_graph) {
            auto& cached = verify_graphs[solo ? B : 0][std::max(0, block_bucket)];
            auto& graph = cached.graph;
            if (!graph.exec || cached.slots != rows.slots || cached.buckets != rows.buckets || cached.mode != mode) {
                if (graph.exec) {
                    CU(cudaGraphExecDestroy(graph.exec));
                    CU(cudaGraphDestroy(graph.graph));
                }
                CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
                launch();
                CU(cudaStreamEndCapture(g_s, &graph.graph));
                CU(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
                cached.slots = rows.slots;
                cached.buckets = rows.buckets;
                cached.mode = mode;
            }
            CU(cudaGraphLaunch(graph.exec, g_s));
        } else launch();
        CU(cudaStreamSynchronize(g_s));
        auto target_token = [&](BatchRequest& r, int t) {
            if (r.sampling.temp <= 0) return W.argmax_h[t];
            auto pick = [&](std::vector<std::pair<float, int>>& candidates) {
                auto p = sample_dist(candidates, r.sampling);
                const int row = t % B;
                if (sample && !r.seeded && row+1 < B)
                    return verify_draft(p, tokens[t+1], mass[(t/B)*(B-1)+row], r.rng);
                return sample_from(p, r.rng);
            };
            std::vector<std::pair<float, int>> candidates;
            const float* logits;
            if (mode == OutMode::kCand) {
                if (block_cands(single ? W.cand_h : buf.cand_h, t, candidates))
                    return pick(candidates);
                CU(cudaMemcpy(W.logits_h, W.logits + (size_t)t * c.n_rows, c.n_vocab * sizeof(float),
                              cudaMemcpyDeviceToHost));
                logits = W.logits_h;
            } else logits = single ? W.logits_h : buf.logits_h + (size_t)t * c.n_vocab;
            candidates.resize(c.n_vocab);
            for (int id = 0; id < c.n_vocab; ++id) candidates[id] = {logits[id], id};
            return pick(candidates);
        };
        std::vector<ik2::draft::ContextWindow> commits;
        for (int first = 0; first < T; first += B) {
            const int s = rows.slots[first];
            const int start = slots[s]->pos;
            int kept = 0;
            if (B > 1 && stats) ++slots[s]->draft_windows;
            while (kept < B) {
                const int token = target_token(*slots[s], first + kept);
                ++kept;
                ++slots[s]->pos;
                if (B > 1 && stats) ++slots[s]->verified_tokens;
                emit(s, token);
                if (!slots[s] || kept == B || token != tokens[first + kept]) break;
            }
            if (drafter && slots[s]) {
                if (kept < B) {
                    const int count = 4 * c.n_visit;
                    k::rollback_block(rollback + s * count, count, kept, g_s);
                }
                commits.push_back({s, first, kept, start});
            }
        }
        if (drafter) ik2::draft::commit_windows(drafter, g_taps, g_taps_ld, commits);
        if (solo && !commits.empty() && solo_ctl[active.front()].adapt) {
            CU(cudaStreamSynchronize(g_s));
            auto& ctl = solo_ctl[active.front()];
            if (single) ctl.took_plain(now() - step_start);
            else ctl.took_spec(B, commits.front().count - 1, now() - step_start);
        }
        std::fflush(stdout);
    }
}
