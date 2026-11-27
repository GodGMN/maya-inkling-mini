#pragma once

#include "ik2_draft.hpp"
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace ik2::draft {
inline int sample_from(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double z = 0;
    for (const auto& e : d)
        if (e.first != skip) z += e.second;
    if (z <= 0) return d[0].first;
    double u = std::uniform_real_distribution<double>(0, z)(rng);
    for (const auto& e : d) {
        if (e.first == skip || e.second <= 0) continue;
        u -= e.second;
        if (u <= 0) return e.first;
    }
    for (auto it = d.rbegin(); it != d.rend(); ++it)
        if (it->first != skip && it->second > 0) return it->first;
    return d[0].first;
}

// The probability missing after accepting min(p,q) is exactly max(p-q,0).
inline int verify_draft(std::vector<std::pair<int, double>>& p, int token, const DraftMass& q,
                        std::mt19937_64& rng) {
    double target_probability = 0, draft_probability = 0;
    for (auto [id, prob] : p) if (id == token) target_probability = prob;
    for (int j = 0; j < kCandidates; ++j) if (q.ids[j] == token) draft_probability = q.probabilities[j];
    if (std::uniform_real_distribution<double>(0, 1)(rng) * draft_probability < target_probability) return token;
    for (auto& [id, prob] : p)
        for (int j = 0; j < kCandidates; ++j)
            if (q.ids[j] == id) { prob = std::max(0.0, prob - q.probabilities[j]); break; }
    return sample_from(p, rng);
}
} // namespace ik2::draft
