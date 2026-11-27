#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ik2 {

// Each reserve is dedicated to one router pick of one physical layer visit.
// A miss temporarily makes that expert resident for the rest of the token.
struct ServedSlots {
    std::vector<int> owner, reserve;
    int hot;

    ServedSlots(int slots, int visits_times_k) : owner(slots, -1), reserve(visits_times_k), hot(slots - visits_times_k) {
        for (int i = 0; i < hot; ++i) owner[i] = i;
        for (int i = 0; i < visits_times_k; ++i) reserve[i] = hot + i;
    }

    void write_table(const uint8_t** table, int experts, const uint8_t* cold, const uint8_t* cache, std::size_t bytes) const {
        for (int e = 0; e < experts; ++e) table[e] = cold + e * bytes;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] >= 0) table[owner[s]] = cache + s * bytes;
        for (int i = 0; i < int(reserve.size()); ++i) table[experts + i] = cache + reserve[i] * bytes;
    }

    void refill(const uint32_t* frequency) {
        std::vector<int> used;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] >= 0) used.push_back(s);
        std::sort(used.begin(), used.end(), [&](int a, int b) {
            if (frequency[owner[a]] != frequency[owner[b]]) return frequency[owner[a]] > frequency[owner[b]];
            return owner[a] < owner[b];
        });
        for (int i = hot; i < int(used.size()); ++i) owner[used[i]] = -1;
        int next = 0;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] < 0) reserve[next++] = s;
    }
};
}
