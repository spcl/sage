#include <cuda.h>
#include <cstdio>
#include <vector>
#include <fstream>
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

void checksum_runner() {
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

    printf("Checksum runner!\n");

    // CUfunction mykernel;
    // CUDA_DRV_CHECK(cuModuleGetFunction(&mykernel, cuModule, "mykernel"));

    // void *args[] = {};
    // CUDA_DRV_CHECK(cuLaunchKernel(mykernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));

    CUDA_DRV_CHECK(cuCtxSynchronize());

    CUDA_DRV_CHECK(cuCtxDestroy(cuContext));
}
