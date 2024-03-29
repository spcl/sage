#pragma once

#include <cuda.h>
#include <cstdio>
#include <cinttypes>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <thread>
#include <random>
#include <cooperative_groups.h>
#include <cooperative_groups/details/helpers.h>


#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            printf("CUDA_ERROR %s:%d %s %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)

#define CUDA_MALLOC(dtype, name, size) dtype* name; CUDA_CHECK(cudaMalloc(&name, size))
#define CUDA_MALLOC_MANAGED(dtype, name, size) dtype* name; CUDA_CHECK(cudaMallocManaged(&name, size))
#define CUDA_MEMCPY(dst, src, size) CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault))
#define SLEEP(seconds) std::this_thread::sleep_for(std::chrono::milliseconds(seconds * 1000))

// #define BATCH 128
// #define IN_FEATURES1 784
// #define OUT_FEATURES1 100
// #define OUT_FEATURES2 10

#define BATCH 128
#define IN_FEATURES1 784
#define OUT_FEATURES1 100
#define OUT_FEATURES2 10

using timer = std::chrono::high_resolution_clock;
inline double seconds(decltype(timer::now() - timer::now()) x) {
    return std::chrono::duration<double>(x).count();
};

typedef void (*SAGEKernelPtr)(void* args);

struct LinearArgs {
    size_t in_features;
    size_t out_features;
    float* weight;
    float* bias;
    size_t batch;
    float* input;
    float* output;
};

struct ReluArgs {
    size_t size;
    float* input;
    float* output;
};

inline void std_mean(double* times, int warmup, int repeats, double& mean, double& std) {
    mean = std::accumulate(&times[warmup], &times[warmup+repeats], 0.0) / repeats;
    auto std_compute = [mean](double acc, double val) {
        return acc + std::pow(val - mean, 2);
    };
    std = std::sqrt(std::accumulate(&times[warmup], &times[warmup+repeats], 0.0, std_compute) / repeats);
}

inline size_t file_size(const char* name) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    return f.tellg();
}

inline void fp_rand(int seed, float* buffer, size_t size) {
    static std::mt19937 generator(seed);
    static std::normal_distribution<double> distribution(0.0, 0.5);
    for (size_t i = 0; i < size; i++) {
        buffer[i] = distribution(generator);
    }
}

template <typename T>
inline std::vector<T> read_file(const char* name) {
    size_t size = file_size(name);
    std::ifstream f(name, std::ios::binary);
    std::vector<T> res(size / sizeof(T));
    f.read((char*)res.data(), size);
    return res;
}

inline void write_file(const char* name, void* buf, size_t size) {
    std::ofstream f(name, std::ios::binary);
    f.write((char*)buf, size);
}