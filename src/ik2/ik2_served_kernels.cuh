#pragma once

__global__ void copy_served_experts(const int* picks, const uint8_t* const* table, const uint8_t* cold,
                                    const uint8_t* const* reserve, size_t bytes, int* records) {
    int k = blockIdx.y, e = picks[k];
    bool miss = table[e] == cold + e * bytes;
    if (!blockIdx.x && !threadIdx.x) records[k] = miss ? ~e : e;
    if (!miss) return;
    auto src = reinterpret_cast<const uint4*>(table[e]);
    auto dst = reinterpret_cast<uint4*>(const_cast<uint8_t*>(reserve[k]));
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

__global__ void publish_served_experts(const int* records, int K, const uint8_t** table,
                                      const uint8_t* const* reserve, const uint8_t** selected) {
    int k = threadIdx.x;
    if (k >= K) return;
    int record = records[k];
    if (record < 0) selected[k] = table[~record] = reserve[k];
}
