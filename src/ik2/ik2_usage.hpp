#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ik2::usage {
namespace fs = std::filesystem;

inline fs::path profile_path(const std::string& model) {
    const char* cache = std::getenv("XDG_CACHE_HOME");
    fs::path dir;
    if (cache && *cache) dir = cache;
#ifdef _WIN32
    else if (const char* local = std::getenv("LOCALAPPDATA")) dir = local;
#else
    else if (const char* home = std::getenv("HOME")) dir = fs::path(home) / ".cache";
#endif
    else return {};
    std::error_code ec;
    const auto path = fs::weakly_canonical(model, ec);
    if (ec) return {};
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char ch : path.u8string()) hash = (hash ^ ch) * 1099511628211ULL;
    return dir / "tintero" / ("expert-usage-" + std::to_string(hash) + ".bin");
}

inline std::vector<uint64_t> read(const fs::path& path, int layers, int experts) {
    if (path.empty()) return {};
    FILE* in = std::fopen(path.string().c_str(), "rb");
    if (!in) return {};
    int32_t hdr[2];
    std::vector<uint64_t> counts;
    if (std::fread(hdr, sizeof(int32_t), 2, in) == 2 && hdr[0] == layers && hdr[1] == experts) {
        counts.resize(size_t(layers) * experts);
        if (std::fread(counts.data(), sizeof(uint64_t), counts.size(), in) != counts.size() ||
            std::fgetc(in) != EOF || std::ferror(in)) counts.clear();
    }
    std::fclose(in);
    std::fprintf(stderr, "expert profile %s: %s\n", counts.empty() ? "invalid" : "loaded", path.string().c_str());
    return counts;
}

inline void write(const fs::path& path, int experts, const std::vector<uint64_t>& counts) {
    if (path.empty() || counts.empty()) return;
    std::error_code ec;
    if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = getpid();
#endif
    const auto tmp = fs::path(path.string() + "." + std::to_string(pid) + ".tmp");
    FILE* out = std::fopen(tmp.string().c_str(), "wb");
    bool ok = false;
    if (out) {
        const int32_t hdr[] = {int32_t(counts.size() / experts), experts};
        ok = std::fwrite(hdr, sizeof(int32_t), 2, out) == 2 &&
             std::fwrite(counts.data(), sizeof(uint64_t), counts.size(), out) == counts.size();
        ok = std::fclose(out) == 0 && ok;
        if (ok) {
#ifdef _WIN32
            ok = MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
            ok = std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
        }
        if (!ok) fs::remove(tmp, ec);
    }
    std::fprintf(stderr, "expert profile %s: %s\n", ok ? "saved" : "save failed", path.string().c_str());
}
}
