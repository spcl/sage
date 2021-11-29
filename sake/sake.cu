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

#define CHALLENGE_SIZE (32)
#define NUM_CHALLENGES (1)

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
    msgs[tid].ptr[0] = tid + '0';

    int curr_id = -1;
    while(true) { // wait for incoming msgs
        if(msgs[tid].id != curr_id) {
            msgs[tid].lock.lock();
            curr_id = msgs[tid].id;
    
            // TODO: process msg
            msgs[tid].ptr[1] = curr_id + '0';
    
            msgs[tid].lock.unlock();

            if (curr_id == 1)
                return;
        }
    }
}

// helper function to transfer the strings to the unified memory used for message passing
void transfer_msg(Message *msg, const char *cts, size_t sz) {
    if (sz > CHALLENGE_SIZE) {
        return;
    }

    msg->lock.lock();
    strncpy(msg->ptr, cts, sz);
    if (sz < CHALLENGE_SIZE) {
        msg->ptr[sz] = '\0';
    }
    msg->size = sz;
    msg->id++;
    msg->lock.unlock();
}

void sake_runner() {
    printf("[G] Running SAKE protocol...\n");

    Message *msgs;
    CUDA_CHECK(cudaMallocHost(&msgs, sizeof(Message)*NUM_CHALLENGES));
    // printf("%lu\n", sizeof(msgs[0]));
    
    char *msg_cts; // msg contents
    CUDA_CHECK(cudaMallocHost(&msg_cts, CHALLENGE_SIZE*NUM_CHALLENGES));

    // link msgs to msg contents
    for (int i=0; i<NUM_CHALLENGES; i++) {
        msgs[i].ptr = msg_cts+i*CHALLENGE_SIZE;
        // printf("%s\n", msgs[i].ptr);
    }

    const char test_msg[] = "asdfasdfasdf";
    // "transfer" msgs
    transfer_msg(&msgs[0], &test_msg[0], strlen(test_msg));
    
    printf("MSG: %s\n", msg_cts);
    sake_test_kernel<<<1,1>>>(msgs);
   
    sleep(1);

    const char test_msg2[] = "hanswurst";
    transfer_msg(&msgs[0], &test_msg2[0], strlen(test_msg2));

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("MSG: %s\n", msg_cts);

    CUDA_CHECK(cudaFreeHost(msgs));
    CUDA_CHECK(cudaFreeHost(msg_cts));
}

// int main() {
//     sake_runner();
//     return 0;
// }
