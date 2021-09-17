// stdlib
#include <cstdio>
#include <vector>
#include <fstream>

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
    printf("kernel_name = %s\n", kernel_name.c_str());
    CUfunction checksum_kernel;
    CUDA_DRV_CHECK(cuModuleGetFunction(&checksum_kernel, cuModule, kernel_name.c_str())); // use checksum function loaded from file

    int multiProcessorCount = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, /* device */ 0));
    int maxRegistersPerMultiprocessor = -1;
    CUDA_DRV_CHECK(cuDeviceGetAttribute(&maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, /* device */ 0));

    int blockSize = -1;
    int numBlocks = -1;
    CUDA_DRV_CHECK(cuOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, checksum_kernel, sharedMemoryPerBlockSize, /* dynamicSMemSize */ 0, /* blockSizeLimit */ 0));

    printf("Suggested number of blocks: %d\n", numBlocks);
    printf("Suggested number of threads per block: %d\n", blockSize);

    CUDA_DRV_CHECK(cuCtxSynchronize());

    CUDA_DRV_CHECK(cuCtxDestroy(cuContext));
}
