// stdlib
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// CUDA related
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRV_CHECK(e) do {\
    CUresult r = e;\
    if (r != CUDA_SUCCESS) {\
       const char* ptr = nullptr;\
       cuGetErrorString(e, &ptr);\
       if (ptr) {\
           printf("CUDA ERROR %s:%d %s %s\n", __FILE__, __LINE__, #e, ptr);\
       } else {\
           printf("CUDA ERROR %s:%d %s INVALID ERROR CODE\n", __FILE__, __LINE__, #e);\
       }\
       exit(1);\
    }\
} while (0)

const uint32_t num_blocks = 4;
const uint32_t data_size = 1<<23;
const uint32_t nonce_size = 2;

size_t sharedMemoryPerBlockSize(int blockSize) {
    return 0;
}

void checksum_runner() {
    printf("[G] Running checksum...\n");

    CUDA_DRV_CHECK(cuInit(0));

    CUcontext cuContext;
    CUDA_DRV_CHECK(cuCtxCreate(&cuContext, /*flags*/ 0, /*device*/ 0));

    CUmodule cuModule;
    CUDA_DRV_CHECK(cuModuleLoad(&cuModule, "checksum/checksum.cubin"));

    std::vector<uint8_t> checksum_code;
    std::ifstream checksum_code_stream("checksum/checksum_function.bin", std::ios::binary);
    char c = -1;
    while (checksum_code_stream.get(c)) {
        checksum_code.push_back(c);
    }

    std::string kernel_name = "checksum_kernel_from_data";
    printf("[G] kernel_name = %s\n", kernel_name.c_str());
    CUfunction checksum_kernel;
    CUDA_DRV_CHECK(cuModuleGetFunction(&checksum_kernel, cuModule, kernel_name.c_str())); // use checksum function loaded from file

    int multiProcessorCount = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, /* device */ 0));
    int maxRegistersPerMultiprocessor = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, /* device */ 0));

    int blockSize = -1;
    int numBlocks = -1;
    CUDA_DRV_CHECK(cuOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, checksum_kernel, sharedMemoryPerBlockSize, /* dynamicSMemSize */ 0, /* blockSizeLimit */ 0));

    printf("[G] Suggested number of blocks: %d\n", numBlocks);
    printf("[G] Suggested number of threads per block: %d\n", blockSize);

    int registersPerThread = 0;
    CUDA_DRV_CHECK(cuFuncGetAttribute(&registersPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, checksum_kernel));
    printf("[G] Number of registers per thread: %d\n", registersPerThread);
    printf("[G] Total number of registers in use: %d out of %d\n", registersPerThread * blockSize * numBlocks, multiProcessorCount * maxRegistersPerMultiprocessor);

    uint32_t host_nonce[nonce_size] = { 1234, 7777 };

    std::vector<uint32_t> host_result(numBlocks);
    CUdeviceptr device_nonce;
    CUdeviceptr device_result;
    CUDA_DRV_CHECK(cuMemAlloc(&device_nonce, sizeof(uint32_t) * nonce_size));
    CUDA_DRV_CHECK(cuMemAlloc(&device_result, sizeof(uint32_t) * numBlocks));

    std::vector<uint32_t> host_data(data_size);
    assert(sizeof(uint32_t) * data_size >= checksum_code.size());
    memcpy(host_data.data(), checksum_code.data(), checksum_code.size());

    CUdeviceptr device_data_ptr;
    CUDA_DRV_CHECK(cuMemAlloc(&device_data_ptr, data_size * sizeof(uint32_t)));
    CUDA_DRV_CHECK(cuMemcpyHtoD(device_data_ptr, host_data.data(), data_size * sizeof(uint32_t)));

    CUDA_DRV_CHECK(cuMemcpyHtoD(device_nonce, &host_nonce, sizeof(uint32_t) * nonce_size));

    CUdeviceptr device_clocks;
    CUDA_DRV_CHECK(cuMemAlloc(&device_clocks, sizeof(uint64_t)));

    // for debug, remove later
    numBlocks = 1;
    blockSize = 1;
    
    void* args[] = {
        (void*) &device_nonce,
        (void*) &device_result,
        (void*) &device_data_ptr,
        (void*) &device_clocks
    };

    printf("[G] Launching checksum kernel...\n");
    CUDA_DRV_CHECK(cuLaunchKernel(checksum_kernel, 
                /* grid size */ numBlocks, 1, 1, /* block size */ blockSize, 1, 1, 
                /* shared mem */ 0, /* stream */ nullptr, args, /* extra params */ nullptr));

    CUDA_DRV_CHECK(cuCtxSynchronize());

    CUDA_DRV_CHECK(cuMemcpyDtoH(&host_result[0], device_result, sizeof(uint32_t) * numBlocks));

    uint64_t host_clocks;
    CUDA_DRV_CHECK(cuMemcpyDtoH(&host_clocks, device_clocks, sizeof(uint64_t)));

    printf("Clocks on GPU %" PRIu64 "\n", host_clocks);

    printf("Checksum computation on GPU:\n");
    for (int i = 0; i < numBlocks; i++) {
        printf("%" PRIx32 " ", host_result[i]);
    }
    printf("\n");

    CUDA_DRV_CHECK(cuCtxDestroy(cuContext));
}
