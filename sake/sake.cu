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

// own
#include "sake.hpp"
#include "sha256.cuh"

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

void sake_runner() {
    printf("[G] Running SAKE protocol...\n");
}