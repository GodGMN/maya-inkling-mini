#pragma once

#include "gguf_reader.hpp"
#include <cmath>
#include <limits>

namespace ik2 {

struct LoraPair {
    const strata::TensorInfo *a = nullptr, *b = nullptr;
    int rank = 0;
    float scale = 0;
};

// Only attention output adapters are wired into every execution path. Reject other
// targets and incomplete pairs before any of their weights reach a device.
inline std::vector<LoraPair> attention_lora_pairs(const strata::GgufFile& file, const std::string& architecture,
                                                  int layers, int input, int output) {
    for (const auto& kv : std::map<std::string, std::string>{{"general.type", "adapter"}, {"adapter.type", "lora"},
                                                              {"general.architecture", architecture}}) {
        const auto* v = file.get(kv.first);
        if (!v || v->type != strata::MetaType::STRING || v->s != kv.second)
            throw std::runtime_error("expected " + kv.first + " = " + kv.second);
    }
    const auto* alpha = file.get("adapter.lora.alpha");
    if (alpha && (alpha->type != strata::MetaType::F32 || !std::isfinite(alpha->f)))
        throw std::runtime_error("expected a finite F32 adapter.lora.alpha");
    std::vector<LoraPair> pairs(layers);
    size_t consumed = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string name = "blk." + std::to_string(layer) + ".attn_output.weight";
        const auto *a = file.find(name + ".lora_a"), *b = file.find(name + ".lora_b");
        if (!a && !b) continue;
        if (!a || !b || a->shape.size() != 2 || b->shape.size() != 2 || a->shape[0] != uint64_t(input) ||
            b->shape[1] != uint64_t(output) || !a->shape[1] || a->shape[1] != b->shape[0] ||
            a->shape[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("missing or incompatible LoRA pair for " + name);
        for (auto* t : {a, b}) {
            if (t->type != 0) throw std::runtime_error("expected F32 weights for " + t->name);
            if (file.data_start() > file.file_size() || t->offset > file.file_size() - file.data_start() ||
                t->elements() > (file.file_size() - file.data_start() - t->offset) / sizeof(float))
                throw std::runtime_error("truncated tensor " + t->name);
        }
        const int rank = static_cast<int>(a->shape[1]);
        pairs[layer] = {a, b, rank, alpha && alpha->f ? static_cast<float>(alpha->f / rank) : 1.0f};
        consumed += 2;
    }
    if (!consumed || consumed != file.tensors().size())
        throw std::runtime_error("expected unique, paired attention output LoRA matrices on existing layers");
    return pairs;
}

}  // namespace ik2
