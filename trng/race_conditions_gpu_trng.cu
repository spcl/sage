#include <cstdio>
#include <fstream>
#include <string>
#include <cooperative_groups.h>
#include <vector>
#include <cassert>
#include <chrono>

namespace cg = cooperative_groups;


#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            printf("CUDA_ERROR %s:%d %s %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)
    
constexpr int bitsPerByte = 8;
constexpr int warpSize = 32;

template <typename T>
__forceinline__
__host__ __device__
volatile T& volatileAccess(T& val) {
    return val;
}

__global__ void myGeneratorKernel(uint8_t* X, uint8_t* R) {
    int threadId = threadIdx.x;
    int threadsPerBlock = blockDim.x;
    int blockId = blockIdx.x;
    int iterations = 128;

    for (int wi = 0; wi < warpSize; wi++) {
        cg::this_grid().sync();
        if (threadId % warpSize == wi) {
            for (int i = 0; i < iterations; i++) {
                volatileAccess(X[threadId]) = 1 - volatileAccess(X[threadId]);
            }
        }
    }
    cg::this_grid().sync();
    if (blockId == 0) {
        if (threadId < threadsPerBlock / bitsPerByte) {
            R[threadId] = 0;
            for (int i = 0; i < bitsPerByte; i++) {
                R[threadId] *= 2;
                R[threadId] += X[bitsPerByte * threadId + i];
            }
        }
    }
}

struct Generator {
    Generator() {
        hR = (uint8_t*) malloc(randomChunkSize);

        assert(hR);

        CUDA_CHECK(cudaMalloc(&dX, workChunkSize));
        CUDA_CHECK(cudaMalloc(&dR, randomChunkSize));

        reset();
    }

    ~Generator() {
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dR));

        free(hR);
    }

    void reset() {
        // shared variables initialization
        CUDA_CHECK(cudaMemset(dX, 0, workChunkSize));
    }

    void generateChunk() {
        reset();

        // kernel
        int numThreadsPerBlock = workChunkSize;
        void* args[] = {
            (void*) &dX,
            (void*) &dR
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)myGeneratorKernel, numBlocks, numThreadsPerBlock, args, 0, 0));

        // data transfer from device to host
        CUDA_CHECK(cudaMemcpy(hR, dR, randomChunkSize, cudaMemcpyDefault));
    }

    void generate(char* buffer, int requiredBytes) {
        for (int startByte = 0; startByte < requiredBytes; startByte += randomChunkSize) {
            int bytesLeft = requiredBytes - startByte;
            int copySize = bytesLeft < randomChunkSize ? bytesLeft : randomChunkSize;
            generateChunk();
            memcpy(buffer + startByte, hR, copySize);
        }
    }

    int numBlocks = 32;
    int workChunkSize = 1024;
    int randomChunkSize = workChunkSize / bitsPerByte;

    uint8_t* hR = nullptr;

    uint8_t* dX = nullptr;
    uint8_t* dR = nullptr;
};

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}
 

int main(int argc, char** argv) {

    std::string filename("random.bin");
    int bytes = 10000000;
    if (argc >= 2) {
        filename = argv[1];
    }
    if (argc >= 3) {
        bytes = atoi(argv[2]);
    }

    std::vector<char> buffer(bytes); 

    Generator gen;

    auto timeStart = hrclock::now();
    gen.generate(buffer.data(), buffer.size());
    auto timeEnd = hrclock::now();

    std::ofstream file(filename, std::ios_base::binary);
    file.write(buffer.data(), bytes);
        
    double elapsedTime = nanoseconds(timeEnd - timeStart).count();
    printf("Bytes: %d, Time: %lg s, Throughput: %lg MB/s\n", bytes, elapsedTime/1e9, bytes/elapsedTime*1e3);
}
