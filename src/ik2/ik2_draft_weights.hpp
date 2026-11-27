#pragma once
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ik2::draft {
// ---- safetensors: the header's tensors (name -> dtype, shape, byte range)
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};
struct StFile {
    std::string path;
    uint64_t data0 = 0;
    std::map<std::string, StTensor> t;
};

inline StFile st_open(const std::string& path) {
    StFile f;
    f.path = std::filesystem::is_directory(path) ? path + "/model.safetensors" : path;
    std::ifstream file(f.path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("drafter: cannot open " + f.path);
    const auto size = file.tellg();
    file.seekg(0);
    uint64_t n = 0;
    if (!file.read(reinterpret_cast<char*>(&n), 8) || !n || n > (64u << 20))
        throw std::runtime_error("drafter: not a safetensors file");
    std::string h(n, ' ');
    if (!file.read(h.data(), n)) throw std::runtime_error("drafter: short header");
    f.data0 = 8 + n;
    // {"name":{"dtype":"BF16","shape":[a,b],"data_offsets":[x,y]}, ...}: a flat scan is enough for this header
    size_t i = h.find('{') + 1;
    while (true) {
        const size_t q0 = h.find('"', i);
        if (q0 == std::string::npos) break;
        const size_t q1 = h.find('"', q0 + 1);
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const size_t ob = h.find('{', q1), oe = h.find('}', ob);
        if (oe == std::string::npos || ob == std::string::npos)
            throw std::runtime_error("drafter: incomplete header entry");
        const std::string body = h.substr(ob, oe - ob);
        i = oe + 1;
        if (name == "__metadata__") continue;
        StTensor t;
        const size_t d0 = body.find("\"dtype\"");
        const size_t d1 = body.find('"', body.find(':', d0) + 1), d2 = body.find('"', d1 + 1);
        t.dtype = body.substr(d1 + 1, d2 - d1 - 1);
        auto ints = [&](const char* key) {
            std::vector<int64_t> v;
            const size_t a = body.find(key);
            const size_t b0 = body.find('[', a), b1 = body.find(']', b0);
            const std::string list = body.substr(b0 + 1, b1 - b0 - 1);
            for (const char* p = list.c_str(); *p;) {
                while (std::isspace(static_cast<unsigned char>(*p)) || *p == ',') ++p;
                if (!*p) break;
                char* end;
                auto value = std::strtoll(p, &end, 10);
                if (p == end || value < 0) throw std::runtime_error("drafter: invalid integer in " + name);
                v.push_back(value);
                p = end;
            }
            return v;
        };
        t.shape = ints("\"shape\"");
        const auto off = ints("\"data_offsets\"");
        if (off.size() != 2) throw std::runtime_error("drafter: bad entry " + name);
        t.begin = (uint64_t) off[0], t.end = (uint64_t) off[1];
        if (t.end < t.begin || t.end > static_cast<uint64_t>(size) - f.data0)
            throw std::runtime_error("drafter: invalid offsets for " + name);
        f.t[name] = t;
    }
    return f;
}

inline std::vector<uint16_t> st_bf16(const StFile& f, const std::string& name, const std::vector<int64_t>& shape) {
    auto it = f.t.find(name);
    if (it == f.t.end()) throw std::runtime_error("drafter: no tensor " + name);
    if (it->second.dtype != "BF16") throw std::runtime_error("drafter: " + name + " is " + it->second.dtype);
    if (it->second.shape != shape) throw std::runtime_error("drafter: " + name + ": unexpected shape");
    uint64_t bytes = sizeof(uint16_t);
    for (auto dim : shape) {
        if (dim <= 0 || uint64_t(dim) > UINT64_MAX / bytes)
            throw std::runtime_error("drafter: invalid shape for " + name);
        bytes *= dim;
    }
    if (bytes != it->second.end - it->second.begin)
        throw std::runtime_error("drafter: wrong byte count for " + name);
    std::vector<uint16_t> v(bytes / 2);
    std::ifstream file(f.path, std::ios::binary);
    file.seekg(f.data0 + it->second.begin);
    if (!file.read(reinterpret_cast<char*>(v.data()), bytes)) throw std::runtime_error("drafter: read failed");
    return v;
}

} // namespace ik2::draft
