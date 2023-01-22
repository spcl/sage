#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cinttypes>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <chrono>
#include <unistd.h>
#pragma GCC target("rdrnd")
#include <immintrin.h>
#include "cuda_src.h"

using timer = std::chrono::high_resolution_clock;
using Time = decltype(timer::now());
double seconds(decltype(timer::now() - timer::now()) x) {
    return std::chrono::duration<double>(x).count();
};

#define CUDA_DRV_CHECK(expr) do { \
    CUresult err = (expr); \
    if (err != 0) { \
        const char* ptr = nullptr; \
        cuGetErrorString(err, &ptr); \
        printf("CUDA_DRV_CHECK detected error %s:%d %s\n", __FILE__, __LINE__, ptr); \
        std::exit(1); \
    } \
} while (0)

extern "C"
void checksum_kernel_reference(State* state, uint32_t* data_ptr, uint32_t grid_size, uint32_t block_size, bool copied_memory);

int main(int argc, char** argv) {
    std::string cubin_name = "cuda_src_ptx.cubin";
    //std::string cubin_name = "cuda_src_ptx_patched.cubin";
    
    std::string kernel_name = "checksum_kernel_from_data";
    //std::string kernel_name = "checksum_kernel";

    //std::string binary_name = "checksum_function_generated.bin";
    std::string binary_name = "checksum_function_extracted.bin";

    bool warmup = false;

    int gridSize = -1;
    int blockSize = -1;

    bool verify = false;

    // when function from file is executed, this flag instructs to use separate buffers for code and data
    bool copy_memory = false;

    // with this flag enabled, verification assumes that 
    // code and data located in the separate buffers
    bool copied_memory = false;

    int opt = 0;
    while ((opt = getopt(argc, argv, "g:t:b:c:k:wvamh")) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s -g gridSize -t blockSize -b foo.bin -c bar.cubin -k kernel_name -w -v -a -m\n", argv[0]);
                printf("\tw = warmup, v = verify\n");
                return 0;
            case 'g':
                gridSize = strtol(optarg, NULL, 10);
                break;
            case 't':
                blockSize = strtol(optarg, NULL, 10);
                break;
            case 'b': 
                binary_name = optarg;
                break;
            case 'c': 
                cubin_name = optarg;
                break;
            case 'k': 
                kernel_name = optarg;
                break;
            case 'w': 
                warmup = true;
                break;
            case 'v': 
                verify = true;
                break;
            case 'a': 
                copy_memory = true;
                break;
            case 'm': 
                copied_memory = true;
                break;
            case '?':
            default:
                return 1;
        }
    }

    CUDA_DRV_CHECK(cuInit(0));

    CUcontext cuContext;
    CUDA_DRV_CHECK(cuCtxCreate(&cuContext, /*flags*/ 0, /*device*/ 0));

    CUmodule cuModule;
    CUDA_DRV_CHECK(cuModuleLoad(&cuModule, cubin_name.c_str()));

    std::vector<uint8_t> checksum_code;
    std::ifstream checksum_code_stream(binary_name, std::ios::binary);
    char c = -1;
    while (checksum_code_stream.get(c)) {
        checksum_code.push_back(c);
    }

    CUfunction checksum_kernel;
    CUDA_DRV_CHECK(cuModuleGetFunction(&checksum_kernel, cuModule, kernel_name.c_str())); // use checksum function loaded from file

    // initialize global data
    CUfunction init_kernel;
    CUDA_DRV_CHECK(cuModuleGetFunction(&init_kernel, cuModule, "init_kernel"));
    void* init_args[] = {};
    CUDA_DRV_CHECK(cuLaunchKernel(init_kernel, 
                        /* grid size */ 1, 1, 1, /* block size */ 1, 1, 1, 
                        /* shared mem */ 0, /* stream */ nullptr, init_args, 0));
    CUDA_DRV_CHECK(cuCtxSynchronize());

    // Initialize state

    std::mt19937 prng;
    int result = 0;
    unsigned int seed;
    
    while (result == 0)
        result = _rdrand32_step(&seed);
    
    prng.seed(seed);

    printf("State:");
    State state;
    state.c = prng();
    printf(" C=0x%08x", state.c);
    for (int i = 0; i < STATE_SIZE; i++) {
        state.d[i] = prng();
        printf(" S%d=0x%08x", i, state.d[i]);
    }
    printf("\n");

    CUdeviceptr device_state;
    CUDA_DRV_CHECK(cuMemAlloc(&device_state, sizeof(State)));
    CUDA_DRV_CHECK(cuMemcpyHtoD(device_state, &state, sizeof(State)));

    CUdeviceptr checksum_clocks;
    CUDA_DRV_CHECK(cuMemAlloc(&checksum_clocks, sizeof(uint64_t)));

    // get grid/block size that achieves 100% occupancy
    if (gridSize <= 0 || blockSize <= 0) {
        CUDA_DRV_CHECK(cuOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, checksum_kernel,
            /*blockSizeToDynamicSMemSize*/ nullptr, /*dynamicSMemSize*/ 0, /*blockSizeLimit*/ 0));
    }

    printf(
        "Configuartion: grid_size=%d block_size=%d binary_name=%s cubin_name=%s kernel_name=%s\n",
        gridSize,
        blockSize,
        binary_name.c_str(),
        cubin_name.c_str(),
        kernel_name.c_str()
    );

    std::vector<uint8_t> host_data(MEM_SIZE * (gridSize + 1));
    // initialize host data with random values to catch possible bugs
    for (int i = 0; i < MEM_SIZE; i++) {
        host_data[i] = (uint8_t)rand();
    }
    assert(MEM_SIZE >= checksum_code.size());
    memcpy(host_data.data(), checksum_code.data(), checksum_code.size());
    for (int blk = 1; blk < (gridSize + 1); blk++) {
        memcpy(host_data.data() + blk * MEM_SIZE, host_data.data(), MEM_SIZE);
    }

    CUdeviceptr device_data_ptr;
    CUDA_DRV_CHECK(cuMemAlloc(&device_data_ptr, MEM_SIZE * (gridSize + 1)));
    CUDA_DRV_CHECK(cuMemcpyHtoD(device_data_ptr, host_data.data(), MEM_SIZE * (gridSize + 1)));

    void* args[] = {
        (void*) &device_state,
        (void*) &device_data_ptr,
        (void*) &copy_memory,
        (void*) &checksum_clocks,
    };

    int numSM = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, /*device*/ 0));
    printf("Number of SMs: %d\n", numSM);

    int clockRate_kHz = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&clockRate_kHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, /*device*/ 0));
    printf("GPU clock rate: %d kHz (unreliable on consumer GPUs: varying clock boost)\n", clockRate_kHz);

    CUdeviceptr sync_bar_ptr;
    size_t sync_bar_size = 0;
    CUDA_DRV_CHECK(cuModuleGetGlobal(&sync_bar_ptr, &sync_bar_size, cuModule, "sync_bar"));
    assert(sync_bar_size == sizeof(unsigned));
    CUDA_DRV_CHECK(cuMemsetD8(sync_bar_ptr, 0, sync_bar_size));

    // https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    // https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
    size_t fmaPerSM = 64;
    size_t aluPerSM = 64;
    size_t totalCores = (fmaPerSM + aluPerSM) * numSM;
    size_t optimalClocks = gridSize * blockSize * EXPECTED_CLOCKS / totalCores;

    int warmup_repeats = warmup ? 20 : 0;
    int repeats = 1;

    double runtime = 0;;

    double real_clock_rate = -1;

    for (int iters = 0; iters < warmup_repeats + repeats; iters++) {
        Time t1 = timer::now();
        CUDA_DRV_CHECK(cuLaunchKernel(checksum_kernel, 
                        /* grid size */ gridSize, 1, 1, /* block size */ blockSize, 1, 1, 
                        /* shared mem */ 0, /* stream */ nullptr, args, 0));

        CUDA_DRV_CHECK(cuCtxSynchronize()); // wait kernel to stop
        Time t2 = timer::now();
        if (iters >= warmup_repeats) {
            runtime += seconds(t2 - t1);
        }
    }
    runtime /= repeats;

    State device_result;
    CUDA_DRV_CHECK(cuMemcpyDtoH(&device_result, device_state, sizeof(State)));

    uint64_t checksum_clocks_host;
    CUDA_DRV_CHECK(cuMemcpyDtoH(&checksum_clocks_host, checksum_clocks, sizeof(uint64_t)));

    printf("Runtime: %lg s\n", runtime);
    printf("GPU clocks: %" PRIu64 "\n", checksum_clocks_host);
    printf("Optimal clocks %zu\n", optimalClocks);
    printf("Observed %.0f %% of peak performance\n", optimalClocks * 100. / checksum_clocks_host);

    if (verify) {
        State reference_result = state;
        Time t1 = timer::now();
        checksum_kernel_reference(&reference_result, (uint32_t*)host_data.data(), gridSize, blockSize, copied_memory);
        printf("Verification on host took %lg s\n", seconds(timer::now() - t1));

        if (device_result.c == reference_result.c) {
            printf("verification SUCCEED dev: %" PRIx32 " host: %" PRIx32 "\n", device_result.c, reference_result.c);
        } else {
            printf("verification FAILED! dev: %" PRIx32 " host: %" PRIx32 "\n", device_result.c, reference_result.c);
        }
    }

    CUDA_DRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
