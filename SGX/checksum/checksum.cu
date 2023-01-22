#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cinttypes>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <unistd.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        printf("CUDA_CHECK detected error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

#define CUDA_DRV_CHECK(expr) do { \
    CUresult err = (expr); \
    if (err != 0) { \
        const char* ptr = nullptr; \
        cuGetErrorString(err, &ptr); \
        printf("CUDA_DRV_CHECK detected error %s:%d %s\n", __FILE__, __LINE__, ptr); \
        std::exit(1); \
    } \
} while (0)


/*
TODO:
* use full register file
* use maximum amount of SMs and threads on them.
* use reduction from CUB
* communicate nonce and checksum result using DMA 
* use generated checksum blocks
* replace library calls: cub reduce by native assembly implementations

NOTES:
* GPU register file is not dynamically addressable
* try both implementations: with function calls ("call" in ptx) and with jumps ("brx.idx" in ptx)
* important thing to use no branching in checksum computations inside warp, this means that warps jump to the same blocks
*/


__device__ void reduce_xor(uint32_t local, uint32_t* result) {
    // each block operates independently
    // warps of the block are reduced to a single warp through a shared memory
    int threadInBlock = threadIdx.x;
    int threadInWarp = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int numWarpsInBlock = blockDim.x / warpSize;
    int blockId = blockIdx.x;

    extern __shared__ uint32_t warp_result[];

    if (warpId >= 1) {
        warp_result[threadInBlock] = local;
    }

    __syncthreads();

    if (warpId == 0) {
        uint32_t v = local;
        for (int i = 1; i < numWarpsInBlock; i++) {
            v ^= warp_result[threadInBlock + i * warpSize];
        }

        // apply warp reduce to get a reduction result in the first thread of each block

        #pragma unroll
        for (int i = 1; i < warpSize; i = i * 2) {
            v ^= __shfl_xor_sync(0xffffffff, v, i);
        }

        if (threadInWarp == 0) {
            // output
            result[blockId] = v;
        }
    }


}

#define NUM_ITERATIONS 8388608

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// operations available in SASS in 7.5 compute capability
// IADD3 a +- b +- c
// FMA a * b +- c
// SHF (b << n) | (a >> (32-n))
// LOP3 a ^&| b ^&| c 
// LOP3 ~a

__device__ void checksum_function(uint32_t* nonce, uint32_t* result, uint32_t* data_ptr) {
    // TODO: make sure that data_ptr points to the function we are executing right now. is it important at all?
    // or maybe currently we don't have any protection from memory copy?

    // TODO: use codegen to insert data_ptr_size constant for faster code
    
    asm volatile(".reg .u32 r_tgt, r_iteration, r_state, r_checksum, r_data;");
    asm volatile(".reg .u32 tmp;");
    asm volatile(".reg .u64 addr, r_data_ptr, tid;");

    asm volatile(".reg .pred p_cond;");

    /* initialize registers */
    asm volatile("ld.u32 r_state, [%0];" :: "l"(nonce));
    asm volatile("ld.u32 r_checksum, [%0 + 4];" :: "l"(nonce));
    asm volatile("mov.u64 r_data_ptr, %0;" :: "l"(data_ptr));
    asm volatile("mov.u32 r_iteration, 0;");
    asm volatile("mov.u32 tmp, %tid.x;");
    asm volatile("cvt.u64.u32 tid, tmp;");

    asm ("ts: .branchtargets BLK0, BLK1, BLK2, BLK3, BEXIT;");

    /* jump to first block based on the checksum value */
    asm volatile("and.b32 r_tgt, r_checksum, 3;"); // tmp = checksum % 4 (num blocks) = b & 3;
    asm volatile("brx.idx r_tgt, ts;");

    /* -------------- BLOCK 0 --------------- */
    asm volatile("BLK0:");

    //{
    //    uint32_t checksum_copy, iteration_copy, state_copy;
    //    asm volatile("mov.u32 %0, r_checksum;": "=r"(checksum_copy));
    //    asm volatile("mov.u32 %0, r_iteration;": "=r"(iteration_copy));
    //    asm volatile("mov.u32 %0, r_state;": "=r"(state_copy));
    //    print_state(10, checksum_copy, iteration_copy, state_copy);
    //}

    /*uint32_t tgt = compute_tgt(++iteration, checksum);*/
    asm volatile("add.u32 r_iteration, r_iteration, 1;"); //iteration += 1;
    asm volatile("setp.eq.u32 p_cond, r_iteration, " STR(NUM_ITERATIONS) ";"); // r_iteration == NUM_ITERATION
    asm volatile("and.b32 tmp, r_checksum, 3;"); // tmp = checksum % 4 (num blocks) = b & 3;
    asm volatile("selp.b32 r_tgt, 4, tmp, p_cond;"); // r_tgt = p_cond ? 64 : tmp

    /* r_data = r_data_ptr + sizeof(uint32_t) * (r_checksum % data_size) */
    asm volatile("and.b32 tmp, r_checksum, 8388607;"); // 1 << 23 - 1
    asm volatile("shl.b32 tmp, tmp, 2;");
    asm volatile("cvt.u64.u32 addr, tmp;");
    asm volatile("add.u64 addr, addr, r_data_ptr;");
    asm volatile("add.u64 addr, addr, tid;");
    asm volatile("ld.u32 r_data, [addr];");

    /* state = simple_prng(state) */
    asm volatile("mul.lo.u32 tmp, r_state, r_state;");
    asm volatile("or.b32 tmp, tmp, 5;");
    asm volatile("and.b32 tmp, tmp, 2147483647;"); // 1 << 31 - 1
    asm volatile("add.u32 r_state, tmp, r_state;");

    /* block-specific computation chunk 1 */
    asm volatile("add.u32 r_checksum, r_checksum, r_data;\n\t"
        "add.u32 r_checksum, r_checksum, r_state;");
    //asm volatile("add.u32 r_checksum, r_checksum, r_data;\n\t"
    //    "sub.u32 r_checksum, r_checksum, r_state;");
    //asm volatile("sub.u32 r_checksum, r_checksum, r_data;\n\t"
    //    "add.u32 r_checksum, r_checksum, r_state;");
    //asm volatile("sub.u32 r_checksum, r_checksum, r_data;\n\t"
    //    "sub.u32 r_checksum, r_checksum, r_state;");
    
    /* insert tgt and current iteration index into checksum with 3-input xor */
    asm volatile("xor.b32 r_checksum, r_checksum, r_tgt;\n\t"
        "xor.b32 r_checksum, r_checksum, r_iteration;");

    /* block-specific computation chunk 2 */
    asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 3;"); // r_checksum = rotate_left(r_checksum, bits = 3)
    //asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 7;"); // r_checksum = rotate_left(r_checksum, bits = 7)
    //asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 13;"); // r_checksum = rotate_left(r_checksum, bits = 13)
    //asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 19;"); // r_checksum = rotate_left(r_checksum, bits = 19)

    /* jump to next block */
    asm ("brx.idx r_tgt, ts;");

    /* -------------- BLOCK 1 --------------- */
    asm volatile("BLK1:");

    /*uint32_t tgt = compute_tgt(++iteration, checksum);*/
    asm volatile("add.u32 r_iteration, r_iteration, 1;"); //iteration += 1;
    asm volatile("setp.eq.u32 p_cond, r_iteration, " STR(NUM_ITERATIONS) ";"); // r_iteration == NUM_ITERATION
    asm volatile("and.b32 tmp, r_checksum, 3;"); // tmp = checksum % 4 (num blocks) = b & 3;
    asm volatile("selp.b32 r_tgt, 4, tmp, p_cond;"); // r_tgt = p_cond ? 64 : tmp

    /* r_data = r_data_ptr + sizeof(uint32_t) * (r_checksum % data_size) */
    asm volatile("and.b32 tmp, r_checksum, 8388607;\n\t" // 1 << 23 - 1
        "shl.b32 tmp, tmp, 2;\n\t"
        "cvt.u64.u32 addr, tmp;\n\t"
        "add.u64 addr, addr, r_data_ptr;\n\t"
        "add.u64 addr, addr, tid;\n\t"
        "ld.u32 r_data, [addr];");

    /* state = simple_prng(state) */
    asm volatile("mul.lo.u32 tmp, r_state, r_state;\n\t"
        "or.b32 tmp, tmp, 5;\n\t"
        "and.b32 tmp, tmp, 2147483647;\n\t" // 1 << 31 - 1
        "add.u32 r_state, tmp, r_state;");

    /* block-specific computation chunk 1 */
    asm volatile("add.u32 r_checksum, r_checksum, r_data;\n\t"
        "sub.u32 r_checksum, r_checksum, r_state;");
    /* insert tgt and current iteration index into checksum with 3-input xor */
    asm volatile("xor.b32 r_checksum, r_checksum, r_tgt;\n\t"
        "xor.b32 r_checksum, r_checksum, r_iteration;");

    /* block-specific computation chunk 2 */
    asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 7;"); // r_checksum = rotate_left(r_checksum, bits = 7)

    /* jump to next block */
    asm volatile("brx.idx r_tgt, ts;");

    /* -------------- BLOCK 2 --------------- */
    asm volatile("BLK2:");

    /*uint32_t tgt = compute_tgt(++iteration, checksum);*/
    asm volatile("add.u32 r_iteration, r_iteration, 1;"); //iteration += 1;
    asm volatile("setp.eq.u32 p_cond, r_iteration, " STR(NUM_ITERATIONS) ";"); // r_iteration == NUM_ITERATION
    asm volatile("and.b32 tmp, r_checksum, 3;"); // tmp = checksum % 4 (num blocks) = b & 3;
    asm volatile("selp.b32 r_tgt, 4, tmp, p_cond;"); // r_tgt = p_cond ? 64 : tmp

    /* r_data = r_data_ptr + sizeof(uint32_t) * (r_checksum % data_size) */
    asm volatile("and.b32 tmp, r_checksum, 8388607;\n\t" // 1 << 23 - 1
        "shl.b32 tmp, tmp, 2;\n\t"
        "cvt.u64.u32 addr, tmp;\n\t"
        "add.u64 addr, addr, r_data_ptr;\n\t"
        "add.u64 addr, addr, tid;\n\t"
        "ld.u32 r_data, [addr];");

    /* state = simple_prng(state) */
    asm volatile("mul.lo.u32 tmp, r_state, r_state;\n\t"
        "or.b32 tmp, tmp, 5;\n\t"
        "and.b32 tmp, tmp, 2147483647;\n\t" // 1 << 31 - 1
        "add.u32 r_state, tmp, r_state;");

    /* block-specific computation chunk 1 */
    asm volatile("sub.u32 r_checksum, r_checksum, r_data;\n\t"
        "add.u32 r_checksum, r_checksum, r_state;");
    
    /* insert tgt and current iteration index into checksum with 3-input xor */
    asm volatile("xor.b32 r_checksum, r_checksum, r_tgt;\n\t"
        "xor.b32 r_checksum, r_checksum, r_iteration;");

    /* block-specific computation chunk 2 */
    asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 13;"); // r_checksum = rotate_left(r_checksum, bits = 13)

    /* jump to next block */
    asm ("brx.idx r_tgt, ts;");

    /* -------------- BLOCK 3 --------------- */
    asm volatile("BLK3:");

    /*uint32_t tgt = compute_tgt(++iteration, checksum);*/
    asm volatile("add.u32 r_iteration, r_iteration, 1;"); //iteration += 1;
    asm volatile("setp.eq.u32 p_cond, r_iteration, " STR(NUM_ITERATIONS) ";"); // r_iteration == NUM_ITERATION
    asm volatile("and.b32 tmp, r_checksum, 3;"); // tmp = checksum % 4 (num blocks) = b & 3;
    asm volatile("selp.b32 r_tgt, 4, tmp, p_cond;"); // r_tgt = p_cond ? 64 : tmp

    /* r_data = r_data_ptr + sizeof(uint32_t) * (r_checksum % data_size) */
    asm volatile("and.b32 tmp, r_checksum, 8388607;\n\t" // 1 << 23 - 1
        "shl.b32 tmp, tmp, 2;\n\t"
        "cvt.u64.u32 addr, tmp;\n\t"
        "add.u64 addr, addr, r_data_ptr;\n\t"
        "add.u64 addr, addr, tid;\n\t"
        "ld.u32 r_data, [addr];");

    /* state = simple_prng(state) */
    asm volatile("mul.lo.u32 tmp, r_state, r_state;\n\t"
        "or.b32 tmp, tmp, 5;\n\t"
        "and.b32 tmp, tmp, 2147483647;\n\t" // 1 << 31 - 1
        "add.u32 r_state, tmp, r_state;");

    /* block-specific computation chunk 1 */
    asm volatile("sub.u32 r_checksum, r_checksum, r_data;\n\t"
        "sub.u32 r_checksum, r_checksum, r_state;");
    
    /* insert tgt and current iteration index into checksum with 3-input xor */
    asm volatile("xor.b32 r_checksum, r_checksum, r_tgt;\n\t"
        "xor.b32 r_checksum, r_checksum, r_iteration;");

    /* block-specific computation chunk 2 */
    asm volatile("shf.l.wrap.b32 r_checksum, r_checksum, r_checksum, 19;"); // r_checksum = rotate_left(r_checksum, bits = 19)

    /* jump to next block */
    asm ("brx.idx r_tgt, ts;");

    /* -------------- BLOCK EXIT --------------- */

    asm volatile("BEXIT:\n\t");

    uint32_t checksum;
    asm volatile("mov.u32 %0, r_checksum;" : "=r"(checksum));

    // reduction of checksum values
    //reduce_xor(checksum, result);
    *result = checksum;
}

using checksum_function_ptr = void (*)(uint32_t*, uint32_t*, uint32_t*);

__device__ checksum_function_ptr checksum_func = nullptr;

extern "C"
__global__ void checksum_kernel(uint32_t* nonce, uint32_t* result, uint32_t* data_ptr, uint64_t* clocks) {
    uint64_t c = clock64();
    // this prevents checksum function to be optimized away so we will be able to extract it from cubin
    checksum_func = checksum_function;
    checksum_func(nonce, result, data_ptr);
    *clocks = clock64() - c;
}


extern "C"
__global__ void checksum_kernel_from_data(uint32_t* nonce, uint32_t* result, uint32_t* data_ptr, uint64_t* clocks) {
    uint64_t c = clock64();
    checksum_function_ptr func = (checksum_function_ptr) data_ptr;
    func(nonce, result, data_ptr);
    *clocks = clock64() - c;
}

const uint32_t num_blocks = 4;
const uint32_t num_iterations = NUM_ITERATIONS;
const uint32_t data_size = 1<<23;
const uint32_t nonce_size = 2;


__host__ __device__
uint32_t rol32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

size_t sharedMemoryPerBlockSize(int blockSize) {
    return 0;
}

__host__ __device__
void verification(uint32_t* nonce, uint32_t* result, uint32_t* data_ptr) {
    uint32_t state = nonce[0];
    uint32_t checksum = nonce[1];

    uint32_t tgt = checksum % 4;
    uint32_t iteration = 0;

    while (true) {
        //printf("CPU block %" PRIu32 ", checksum %" PRIx32 ", iteration %" PRIu32 ", state %" PRIu32 "\n", (10 * (tgt + 1)) + 0, checksum, iteration, state);

        iteration += 1;
        uint32_t next_tgt = checksum % num_blocks;
        if (iteration == num_iterations) { next_tgt = num_blocks; }
        
        // TODO: add tid to computation!
        uint32_t data = data_ptr[checksum % data_size];

        state = state + ((state * state) | 5u) % ((1u << 31u));

        //printf("CPU block %" PRIu32 ", checksum %" PRIx32 ", iteration %" PRIu32 ", state %" PRIu32 "\n", (10 * (tgt + 1)) + 1, checksum, iteration, state);

        switch (tgt) {
            case 0: checksum = checksum + data + state; break;
            case 1: checksum = checksum + data - state; break;
            case 2: checksum = checksum - data + state; break;
            case 3: checksum = checksum - data - state; break;
        }
        //printf("CPU block %" PRIu32 ", checksum %" PRIx32 ", iteration %" PRIu32 ", state %" PRIu32 "\n", (10 * (tgt + 1)) + 2, checksum, iteration, state);

        checksum = checksum ^ next_tgt ^ iteration;
        //printf("CPU block %" PRIu32 ", checksum %" PRIx32 ", iteration %" PRIu32 ", state %" PRIu32 "\n", (10 * (tgt + 1)) + 3, checksum, iteration, state);

        switch (tgt) {
            case 0: checksum = rol32(checksum, 3); break;
            case 1: checksum = rol32(checksum, 7); break;
            case 2: checksum = rol32(checksum, 13); break;
            case 3: checksum = rol32(checksum, 19); break;
        }
        //printf("CPU block %" PRIu32 ", checksum %" PRIx32 ", iteration %" PRIu32 ", state %" PRIu32 "\n", (10 * (tgt + 1)) + 4, checksum, iteration, state);

        tgt = next_tgt;
        if (next_tgt == num_blocks) {
            break;
        }
    }

    *result = checksum;
    
}

extern "C" __global__
void baseline(uint32_t* nonce, uint32_t* result, uint32_t* data_ptr, uint64_t* clocks) {
    uint64_t c = clock64();
    verification(nonce, result, data_ptr);
    *clocks = clock64() - c;
}

int main(int argc, char** argv) {
//void init_checksum() {
    std::string kernel_name = "checksum_kernel_from_data";


    bool warmup = false;
    /*
    int opt = 0;
    while ((opt = getopt(argc, argv, "k:w")) != -1) {
        switch (opt) {
            case 'k': 
                kernel_name = optarg;
                break;
            case 'w': 
                warmup = true;
                break;
            case '?':
            default:
                return 1;
        }
    }
    */

    CUDA_DRV_CHECK(cuInit(0));

    CUcontext cuContext;
    CUDA_DRV_CHECK(cuCtxCreate(&cuContext, /*flags*/ 0, /*device*/ 0));

    CUmodule cuModule;
    CUDA_DRV_CHECK(cuModuleLoad(&cuModule, "checksum.cubin"));

    std::vector<uint8_t> checksum_code;
    std::ifstream checksum_code_stream("checksum_function.bin", std::ios::binary);
    char c = -1;
    while (checksum_code_stream.get(c)) {
        checksum_code.push_back(c);
    }


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

    // alternative way to compute max number of threads per block with fixed block size
    //int blockSize = 512;
    //int numBlocks = -1;
    //CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, checksum, blockSize, /* required shared memory */ 0));

    printf("Suggested number of blocks: %d\n", numBlocks);
    printf("Suggested number of threads per block: %d\n", blockSize);

    int registersPerThread = 0;
    CUDA_DRV_CHECK(cuFuncGetAttribute(&registersPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, checksum_kernel));
    // If this value is different from nvcc compiler output with "--ptxas-options=-v" option, then
    // probably code is not compiled with exact compute capability of device which runs binary.
    // In this case, the number printed here is correct one.
    printf("Number of registers per thread: %d\n", registersPerThread);
    printf("Total number of registers in use: %d out of %d\n", registersPerThread * blockSize * numBlocks, multiProcessorCount * maxRegistersPerMultiprocessor);

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

    void* args[] = {
        (void*) &device_nonce,
        (void*) &device_result,
        (void*) &device_data_ptr,
        (void*) &device_clocks
    };


    // for debug, remove later
    numBlocks = 1;
    blockSize = 1;

    if (warmup) {
        CUDA_DRV_CHECK(cuLaunchKernel(checksum_kernel, 
                    /* grid size */ numBlocks, 1, 1, /* block size */ blockSize, 1, 1, 
                    /* shared mem */ 0, /* stream */ nullptr, args, /* extra params */ nullptr));
    }
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

    std::vector<uint32_t> reference_result(numBlocks);
    verification(host_nonce, reference_result.data(), host_data.data());

    printf("Checksum computation on CPU:\n");
    for (int i = 0; i < numBlocks; i++) {
        printf("%" PRIx32 " ", reference_result[i]);
    }
    printf("\n");

    bool ok = true;
    for (int i = 0; i < numBlocks; i++) {
        if (host_result[i] != reference_result[i]) {
            ok = false;           
        }
    }
    if (ok) {
        printf("verification succeed\n");
    } else {
        printf("verification failed\n");
    }

    CUDA_DRV_CHECK(cuCtxDestroy(cuContext));

    //return 0;
}
