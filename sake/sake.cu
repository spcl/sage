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

__device__ void d_print_hex(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        if (i > 0)
            printf(":");
        printf("%02X", buf[i]);
    }
    printf("\n");
}

// helper function to transfer the strings to the unified memory used for message passing
// NOTE: care for buffer overflow
__device__ void transfer_msg(Message *msg, const char *buf, size_t buf_size) {
    msg->lock.lock();
    for (size_t i=0; i<buf_size; i++) { // no strncpy on cuda :/
       msg->ptr[i] = buf[i]; 
    }
    msg->ptr[buf_size] = '\0';
    msg->size = buf_size;
    msg->id++;
    msg->lock.unlock();
}

__global__ void sake_test_kernel(volatile uint8_t* run, Message* msgs, Message* rsps) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    volatile int curr_id = -1;
    while(*run) { // wait for incoming msgs
        //printf("Msg id: %d, curr id: %d\n", msgs[tid].id, curr_id);
        //d_print_msg(&msgs[0]);

        if(msgs[tid].id != curr_id) {
            msgs[tid].lock.lock();
            curr_id = msgs[tid].id;

            // TODO: call checksum function
            // printf("GPU kernel:\t");
            // d_print_hex((uint8_t*)msgs[tid].ptr, msgs[tid].size);

            // transfer response back to app
            // TODO: move after unlock once not dependent on msg anymore
            transfer_msg(&rsps[tid], msgs[tid].ptr, 16);

            msgs[tid].lock.unlock();
        }
    }

    // printf("kernel msg id %d, curr id %d\n", msgs[tid].id, curr_id);
    printf("KERNEL DONE\n");
}

void sake_malloc(uint8_t** run, Message** msgs, Message** rsps) {
    CUDA_CHECK(cudaMallocHost(run, 1));

    char* msg_buf;
    CUDA_CHECK(cudaMallocHost(&msg_buf, CHALLENGE_SIZE*NUM_CHALLENGES));
    CUDA_CHECK(cudaMallocHost(msgs, sizeof(Message)*NUM_CHALLENGES));

    char* rsp_buf;
    CUDA_CHECK(cudaMallocHost(&rsp_buf, CHALLENGE_SIZE*NUM_CHALLENGES));
    CUDA_CHECK(cudaMallocHost(rsps, sizeof(Message)*NUM_CHALLENGES));
    
    // link msgs struct to msg buffer
    for (int i=0; i<NUM_CHALLENGES; i++) {
        (*msgs)[i].ptr = msg_buf+i*CHALLENGE_SIZE;
        (*msgs)[i].id = 0;

        (*rsps)[i].ptr = rsp_buf+i*CHALLENGE_SIZE;
        (*rsps)[i].id = 0;
    }
}

void sake_free(uint8_t* run, Message* msgs, Message* rsps) {
    CUDA_CHECK(cudaFreeHost(run));
    CUDA_CHECK(cudaFreeHost(msgs[0].ptr));
    CUDA_CHECK(cudaFreeHost(msgs));
    CUDA_CHECK(cudaFreeHost(rsps[0].ptr));
    CUDA_CHECK(cudaFreeHost(rsps));
}

void sake_runner(uint8_t* run, Message* msgs, Message* rsps) {
    printf("[D] Running SAKE protocol...\n");
    *run = 1;

    sake_test_kernel<<<1,1>>>(run, msgs, rsps);
}

// nvcc -o test sake.cu sha256.cu -arch=sm_61 -lcuda
// int main() {
//     uint8_t* run;
//     Message* msgs;
//     sake_malloc(&run, &msgs);
//     sake_runner(run, msgs);
//     sleep(2);
//     *run = 0;
//     printf("[D] Stop running kernel.\n");
//     sake_free(run, msgs);
//     return 0;
// }
