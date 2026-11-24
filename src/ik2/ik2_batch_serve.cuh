// One line from the Maya frontend owns one slot until DONE. Only the scheduler touches GPU state.
struct BatchRequest {
    int rid = 0, limit = 0, generated = 0, pos = 0, token = 0;
    std::vector<int> ids, stops;
    std::string image_path;
    SampleCfg sampling;
    std::mt19937_64 rng;
    double start = 0, decode_start = 0;
    bool ready = false;

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

void serve_sequences(Model& M, int width, int ctx, int chunk, int image_token, std::vector<int> eos) {
    const Config& c = M.c;
    Work& W = dev_work(0);
    if (!g_fast || g_managed_experts || host_plan.forced || !M.out_w.q8 ||
        !std::all_of(M.L.begin(), M.L.end(), [](const Layer& l) {
            return l.wo.q8 && (l.sliding ? l.wqkvrg.q8 != nullptr : l.wqrg.q8 && l.wk.q8 && l.wv.q8);
        })) die("IK2_BATCH requires resident experts and planar Q8 projections");
    const auto base = M.V;
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
    BlockBufs buf;
    buf.emb_h = W.emb_h; buf.pos_h = W.pos_h; buf.lp_msg = W.lp_msg_all;
    buf.blob = W.blob; buf.tok = W.tok; buf.inv = W.inv;
    buf.sg = W.sg; buf.su = W.su; buf.wt = W.wt;
    CU(cudaHostAlloc(&buf.logits_h, (size_t)width * c.n_vocab * sizeof(float), cudaHostAllocPortable));
    const size_t candidate_ints = (size_t)width * (2 + 2 * kCandCap);
    buf.cand_d = dalloc<int>(candidate_ints);
    CU(cudaHostAlloc(&buf.cand_h, candidate_ints * sizeof(int), cudaHostAllocPortable));
    const bool use_candidates = !getenv("IK2_CAND") || std::atoi(getenv("IK2_CAND"));
    SequenceRows rows;
    rows.device_slots = dalloc<int>(width);
    FastGraph graph;
    std::vector<int> graph_slots, graph_buckets;
    OutMode graph_mode = OutMode::kNone;
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
    auto finish = [&](int s, const char* reason) {
        BatchRequest& r = *slots[s];
        const double t = now(), start = r.ready ? r.decode_start : t;
        std::printf("DONE %d %d %zu %.1f %.1f %s 0 0 0\n", r.rid, r.generated, r.ids.size(),
                    (start - r.start) * 1e3, (t - start) * 1e3, reason);
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
            else if (key == "seed" && std::atoll(value) >= 0) r.rng.seed(std::atoll(value));
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
        if (!valid || !r.rid || r.limit < 0 || r.ids.size() + r.limit > (size_t)ctx || free == slots.end()) {
            std::printf("ERR %d %s\n", r.rid, free == slots.end() ? "all slots busy" : "invalid prompt or context length");
            return;
        }
        r.start = now();
        *free = std::move(request);
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
            if (idle) inbox->changed.wait(lock, [&] { return inbox->eof || !inbox->lines.empty(); });
            if (idle && inbox->eof && inbox->lines.empty()) break;
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
                for (int s = 0; s < width; ++s)
                    if (slots[s] && (!tagged || slots[s]->rid == rid)) finish(s, "cancel");
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
            forward(M, r.ids.data() + r.pos, n, r.pos, false, last ? OutMode::kLogits : OutMode::kNone,
                    {}, logits, nullptr, t_exp);
            g_graph = graphs;
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
        for (int s = 0; s < width; ++s) {
            if (!slots[s] || !slots[s]->ready) continue;
            const auto& r = *slots[s];
            const size_t t = rows.slots.size();
            rows.slots.push_back(s);
            rows.buckets.push_back(decode_bucket(r.pos));
            if (r.sampling.temp > 0) mode = use_candidates ? OutMode::kCand : OutMode::kLogits;
            const size_t rb = (size_t)c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
            k::decode_host(M.embd_type, M.embd + (size_t)r.token * rb, c.C, W.emb_h + t * c.C);
            W.pos_h[t] = r.pos;
        }
        if (rows.slots.empty()) continue;
        M.V = base;
        const int T = rows.slots.size();
        if (T == 1) bind_slot(rows.slots.front(), false);
        auto launch = [&] {
            if (T == 1) fast_token_launches(M, W, mode, rows.buckets.front());
            else fast_block_launches(M, W, T, mode, buf, &rows);
        };
        CU(cudaMemcpyAsync(rows.device_slots, rows.slots.data(), T * sizeof(int), cudaMemcpyHostToDevice, g_s));
        if (g_abl_tokens >= 0 && g_abl_srow && T <= kAblRows) {   // each row's ablation strength by its generated tokens
            for (int t = 0; t < T; ++t) {
                const BatchRequest& r = *slots[rows.slots[t]];
                g_abl_srow_h[t] = r.pos - (int) r.ids.size() < g_abl_tokens ? 1.0f : 0.0f;
            }
            CU(cudaMemcpyAsync(g_abl_srow, g_abl_srow_h, T * sizeof(float), cudaMemcpyHostToDevice, g_s));
        }
        if (g_graph) {
            if (!graph.exec || graph_slots != rows.slots || graph_buckets != rows.buckets || graph_mode != mode) {
                if (graph.exec) {
                    CU(cudaGraphExecDestroy(graph.exec));
                    CU(cudaGraphDestroy(graph.graph));
                }
                CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
                launch();
                CU(cudaStreamEndCapture(g_s, &graph.graph));
                CU(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
                graph_slots = rows.slots;
                graph_buckets = rows.buckets;
                graph_mode = mode;
            }
            CU(cudaGraphLaunch(graph.exec, g_s));
        } else launch();
        CU(cudaStreamSynchronize(g_s));
        for (int t = 0; t < T; ++t) {
            const int s = rows.slots[t];
            BatchRequest& r = *slots[s];
            ++r.pos;
            if (r.sampling.temp <= 0) {
                emit(s, W.argmax_h[t]);
                continue;
            }
            if (mode == OutMode::kCand) {
                std::vector<std::pair<float, int>> candidates;
                if (block_cands(T == 1 ? W.cand_h : buf.cand_h, t, candidates)) {
                    emit(s, sample_from(sample_dist(candidates, r.sampling), r.rng));
                    continue;
                }
                CU(cudaMemcpy(W.logits_h, W.logits + (size_t)t * c.n_rows, c.n_vocab * sizeof(float),
                              cudaMemcpyDeviceToHost));
                emit(s, r.pick(W.logits_h, c.n_vocab));
            } else {
                const float* logits = T == 1 ? W.logits_h : buf.logits_h + (size_t)t * c.n_vocab;
                emit(s, r.pick(logits, c.n_vocab));
            }
        }
        std::fflush(stdout);
    }
}
