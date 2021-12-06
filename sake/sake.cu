// stdlib
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <unistd.h>

// CUDA related
#include <cuda.h>
#include <cuda_runtime.h>

// own
#include "../common/cuda_mem.cuh"
#include "sake.hpp"
#include "sha256.cuh"

#define CHALLENGE_SIZE 32
#define NUM_CHALLENGES 1

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        printf("CUDA_CHECK detected error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

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

__global__ void sake_test_kernel(Message* msgs) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int curr_id = -1;
    volatile int infinity = 1;
    while(infinity) { // wait for incoming msgs
        if(msgs[tid].id != curr_id) {
            msgs[tid].lock.lock();
            curr_id = msgs[tid].id;
    
            // TODO: process msg
            msgs[tid].ptr[0] = tid + '0';
    
            msgs[tid].lock.unlock();

            // if (curr_id == 1)
            //     return;
        }
    }
}

// helper function to transfer the strings to the unified memory used for message passing
// NOTE: care for buffer overflow
void transfer_msg(Message *msg, const char *buf, size_t buf_size) {
    msg->lock.lock();
    strncpy(msg->ptr, buf, buf_size);
    msg->ptr[buf_size] = '\0';
    msg->size = buf_size;
    msg->id++;
    msg->lock.unlock();
}

void sake_runner(Message** msgs) {
    printf("[D] Running SAKE protocol...\n");
    
    char* msg_buf;
    CUDA_CHECK(cudaMallocHost(&msg_buf, CHALLENGE_SIZE*NUM_CHALLENGES));

    // Message *msgs;
    CUDA_CHECK(cudaMallocHost(msgs, sizeof(Message)*NUM_CHALLENGES));
    
    // link msgs struct to msg buffer
    for (int i=0; i<NUM_CHALLENGES; i++)
        (*msgs)[i].ptr = msg_buf+i*CHALLENGE_SIZE;

    sake_test_kernel<<<1,1>>>(*msgs);

    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaFreeHost(msgs));
    // CUDA_CHECK(cudaFreeHost(*msg_buf));
}

// int main() {
//     sake_runner();
//     return 0;
// }
