#include "mlp.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


#define SECRET 0xBA0BAB

#define MAX_MESSAGE_DATA 4
#define MAX_MESSAGE_QUEUE 16

struct Message {
    uint64_t data[MAX_MESSAGE_DATA];
};

#define MESSAGE_EXIT 0
#define MESSAGE_RUN_KERNEL 1
#define MESSAGE_COPY_H2D 2
#define MESSAGE_COPY_D2H 3
#define MESSAGE_ALLOC 4

//__device__ unsigned int grid_barrier;

struct CommChannel {
    volatile int device_ready = 0;

    volatile int queue_first = 0;
    volatile int queue_last = 0;
    volatile Message message_queue[MAX_MESSAGE_QUEUE];

    void* allocated_mem;  // device writes pointer address returned from malloc
    
    void* tmp_buf;  // here we store pointer of buffer for host to device transfers
    size_t tmp_buf_size;

    cudaStream_t stream;
    CommChannel() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        tmp_buff_realloc(128ull << 20);
    }

    void tmp_buff_realloc(size_t size) {
        tmp_buf_size = size;
        CUDA_CHECK(cudaFreeAsync(tmp_buf, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMallocAsync(&tmp_buf, tmp_buf_size, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void wait_while_queue_is_full() {
        while ((queue_last + 1) % MAX_MESSAGE_QUEUE == queue_first) {
            // device queue is full
        }
    }

    void commit_message_to_queue() {
        __sync_synchronize();
        queue_last = (queue_last + 1) % MAX_MESSAGE_QUEUE;
    }

    void synchronize() volatile {
        __sync_synchronize();
        while (queue_last != queue_first) {
            // wait until queue is not empty
        }
    }

    void* alloc(size_t size) {
        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = (uint64_t)MESSAGE_ALLOC;
        message_queue[queue_last].data[1] = (uint64_t)size;
        commit_message_to_queue();
        synchronize();
        return allocated_mem;
    }

    void submit_kernel(void* func, void* args) {
        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = ((uint64_t)MESSAGE_RUN_KERNEL) | ((uint64_t)func);
        message_queue[queue_last].data[1] = (uint64_t)args;
        commit_message_to_queue();
    }

    void copy_h2d(void* dst, void* src, size_t size) {
        size_t aligned_size = ((size - 1) / sizeof(uint64_t) + 1) * sizeof(uint64_t);
        // allocate mem on device        
        if (aligned_size > tmp_buf_size) tmp_buff_realloc(aligned_size);
        CUDA_CHECK(cudaMemcpyAsync(tmp_buf, src, size, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = ((uint64_t)MESSAGE_COPY_H2D) | ((uint64_t)dst);
        message_queue[queue_last].data[1] = (uint64_t)tmp_buf;
        message_queue[queue_last].data[2] = (uint64_t)aligned_size;
        commit_message_to_queue();

        synchronize();
    }

    void copy_d2h(void* dst, void* src, size_t size) {
        size_t aligned_size = ((size - 1) / sizeof(uint64_t) + 1) * sizeof(uint64_t);
        // allocate mem on device
        if (aligned_size > tmp_buf_size) tmp_buff_realloc(aligned_size);

        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = ((uint64_t)MESSAGE_COPY_D2H) | ((uint64_t)tmp_buf);
        message_queue[queue_last].data[1] = (uint64_t)src;
        message_queue[queue_last].data[2] = (uint64_t)aligned_size;
        commit_message_to_queue();

        synchronize();

        CUDA_CHECK(cudaMemcpyAsync(dst, tmp_buf, size, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void shutdown() {
        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = MESSAGE_EXIT;
        commit_message_to_queue();
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
};

__device__ void fastcopy(void* dst, void* src, size_t size) {
    size_t base = threadIdx.x + blockDim.x * blockIdx.x;
    size_t jump = blockDim.x * gridDim.x;
    
    auto dst64 = (uint64_t*)dst;
    auto src64 = (uint64_t*)src;
    for (size_t i = base; i < size / sizeof(uint64_t); i += jump) {
        dst64[i] = src64[i];
    }
    
}

__global__ void kernel_launcher(volatile CommChannel* comm_channel) {
    // assume here we finished running checksum function
    // and established secret key

    cg::this_grid().sync();
    //cooperative_groups::details::grid::sync(&grid_barrier);
    if (threadIdx.x == 0 && blockIdx.x == 0) comm_channel->device_ready = 1;

    // threads don't have to know how much progress is done by others
    int local_queue_first = 0;

    while (1) {
        if (threadIdx.x + blockIdx.x == 0) {
            while (comm_channel->queue_last == local_queue_first) {}
        }
        //cooperative_groups::details::grid::sync(&grid_barrier);
        cg::this_grid().sync();

        // removed volatile, as it has huge impact on performance. Result is still correct, so I guess it is fine.
        Message* m = (Message*) &comm_channel->message_queue[local_queue_first];

        uint64_t* d = m->data;
        uint64_t arg0 = d[0];
        uint64_t arg1 = d[1];

        uint64_t msg = arg0 & 7ull;
        arg0 = arg0 & (~7ull);

        if (msg == MESSAGE_RUN_KERNEL) {
            SAGEKernelPtr kernel = (SAGEKernelPtr)arg0;
            void* args = (void*)arg1;
            kernel(args);
        } else if (msg == MESSAGE_COPY_H2D) {
            uint64_t arg2 = m->data[2];
            char* dst = (char*) arg0;
            char* src = (char*) arg1;
            size_t size = arg2;
            fastcopy(dst, src, size);
        } else if (msg == MESSAGE_COPY_D2H) {
            uint64_t arg2 = m->data[2];
            char* dst = (char*) arg0;
            char* src = (char*) arg1;
            size_t size = arg2;
            fastcopy(dst, src, size);
        } else if (msg == MESSAGE_ALLOC) {
            if (threadIdx.x + blockIdx.x == 0) {
                comm_channel->allocated_mem = malloc(((arg1 - 1) / sizeof(uint64_t) + 1) * sizeof(uint64_t));
            }
        } else if (msg == MESSAGE_EXIT) {
            if (threadIdx.x + blockIdx.x == 0) {
                printf("Exit message\n");
            } 
            return;
        } else {
            if (threadIdx.x + blockIdx.x == 0) printf("Unknown message\n");
        }

        local_queue_first = (local_queue_first + 1) % MAX_MESSAGE_QUEUE;
        if (threadIdx.x + blockIdx.x == 0) {
            comm_channel->queue_first = local_queue_first;
        }
    }
}

int main(int argc, char** argv) {
    // assume this code runs inside SGX

    int warmup = 3;
    int repeats = 10;
    int batch_scale = 1;
    int copy_repeats = 1;
    int relu_repeats = 1;
    int inidividual_sync = 0;

    if (argc != 7) {
        printf("Use: %s warmup repeats batch_scale kernel_repeats relu_repeats inidividual_sync\n", argv[0]);
        printf("Example: %s %d %d %d %d %d %d\n", argv[0], warmup, repeats, batch_scale, copy_repeats, relu_repeats, inidividual_sync);
        return 1;
    } else {
        warmup = atoi(argv[1]);
        repeats = atoi(argv[2]);
        batch_scale = atoi(argv[3]);
        copy_repeats = atoi(argv[4]);
        relu_repeats = atoi(argv[5]);
        inidividual_sync = atoi(argv[6]);
    }

    int batch = batch_scale * BATCH;

    size_t heap_size = 4ull << 30;  // 4GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));

    std::vector<uint8_t> linear_code = read_file<uint8_t>("linear.bin");
    printf("Linear code size %zu\n", linear_code.size());

    std::vector<uint8_t> relu_code = read_file<uint8_t>("relu.bin");
    printf("Relu code size %zu\n", relu_code.size());

    std::vector<float> host_weight1(OUT_FEATURES1 * IN_FEATURES1);
    std::vector<float> host_bias1(OUT_FEATURES1);
    std::vector<float> host_weight2(OUT_FEATURES2 * OUT_FEATURES1);
    std::vector<float> host_bias2(OUT_FEATURES2);

    std::vector<float> host_input(batch * IN_FEATURES1);
    std::vector<float> host_internal(batch * OUT_FEATURES1);
    std::vector<float> host_output(batch * OUT_FEATURES2);

    // create communication channel with device
    CommChannel* comm_channel;
    CUDA_CHECK(cudaMallocHost(&comm_channel, sizeof(CommChannel)));
    new (comm_channel) CommChannel();
    volatile CommChannel* vcomm_channel = comm_channel;

    CUDA_CHECK(cudaDeviceSynchronize());

    int grid, block;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid, &block, kernel_launcher));
    printf("grid %d block %d\n", grid, block);

    // run checksum verification and key establishment
    void* args[] = {
        (void*) &comm_channel
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_launcher, grid, block, args, 0, 0));
    //kernel_launcher<<<grid, block>>>(comm_channel);
    CUDA_CHECK(cudaPeekAtLastError());

    printf("waiting for device...\n");
    while (!vcomm_channel->device_ready) {}
    printf("waiting for device... Done\n");

    auto dev_weight1 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES1 * IN_FEATURES1);
    auto dev_bias1 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES1);
    auto dev_weight2 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES2 * OUT_FEATURES1);
    auto dev_bias2 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES2);

    auto dev_input = (float*) comm_channel->alloc(sizeof(float) * batch * IN_FEATURES1);
    auto dev_internal = (float*) comm_channel->alloc(sizeof(float) * batch * OUT_FEATURES1);
    auto dev_output = (float*) comm_channel->alloc(sizeof(float) * batch * OUT_FEATURES2);

    LinearArgs l1_args; 
    l1_args.in_features = IN_FEATURES1;
    l1_args.out_features = OUT_FEATURES1;
    l1_args.weight = dev_weight1;
    l1_args.bias = dev_bias1;
    l1_args.batch = batch;
    l1_args.input = dev_input;
    l1_args.output = dev_internal;

    ReluArgs relu_args;
    relu_args.size = batch * OUT_FEATURES1;
    relu_args.input = dev_internal;
    relu_args.output = dev_internal;

    LinearArgs l2_args;
    l2_args.in_features = OUT_FEATURES1;
    l2_args.out_features = OUT_FEATURES2;
    l2_args.weight = dev_weight2;
    l2_args.bias = dev_bias2;
    l2_args.batch = batch;
    l2_args.input = dev_internal;
    l2_args.output = dev_output;

    auto dev_l1_args = (LinearArgs*)comm_channel->alloc(sizeof(LinearArgs));
    auto dev_relu_args = (ReluArgs*)comm_channel->alloc(sizeof(ReluArgs));
    auto dev_l2_args = (LinearArgs*)comm_channel->alloc(sizeof(LinearArgs));

    comm_channel->copy_h2d(dev_l1_args, &l1_args, sizeof(LinearArgs));
    comm_channel->copy_h2d(dev_relu_args, &relu_args, sizeof(ReluArgs));
    comm_channel->copy_h2d(dev_l2_args, &l2_args, sizeof(LinearArgs));

    auto dev_linear_code = (uint8_t*) comm_channel->alloc(linear_code.size());
    auto dev_relu_code = (uint8_t*) comm_channel->alloc(relu_code.size());

    comm_channel->copy_h2d(dev_linear_code, linear_code.data(), linear_code.size());
    comm_channel->copy_h2d(dev_relu_code, relu_code.data(), relu_code.size());

    fp_rand(101, host_weight1.data(), OUT_FEATURES1 * IN_FEATURES1);
    fp_rand(102, host_bias1.data(), OUT_FEATURES1);
    fp_rand(103, host_weight2.data(), OUT_FEATURES2 * OUT_FEATURES1);
    fp_rand(104, host_bias2.data(), OUT_FEATURES2);
    fp_rand(105, host_input.data(), batch * IN_FEATURES1);

    // copy encrypted model to device
    comm_channel->copy_h2d(dev_weight1, host_weight1.data(), host_weight1.size() * sizeof(float));
    comm_channel->copy_h2d(dev_bias1, host_bias1.data(), host_bias1.size() * sizeof(float));
    comm_channel->copy_h2d(dev_weight2, host_weight2.data(), host_weight2.size() * sizeof(float));
    comm_channel->copy_h2d(dev_bias2, host_bias2.data(), host_bias2.size() * sizeof(float));

    printf("Start benchmarking...\n");

    std::vector<double> times;
    std::vector<double> times_in, times_out, times_k1, times_k2, times_k3;
    for (int r = 0; r < warmup + repeats; r++) {
        auto t1 = timer::now();

        printf("Copy input to device...\n");
        for (int i = 0; i < copy_repeats; i++) {
            comm_channel->copy_h2d(dev_input, host_input.data(), host_input.size() * sizeof(float));
        }
        printf("Copy input to device...Done\n");
        auto t_input = timer::now();

        printf("Run kernels...\n");
        comm_channel->submit_kernel(dev_linear_code, dev_l1_args);
        if (inidividual_sync) comm_channel->synchronize();
        auto t_k1 = timer::now();
        printf("kernel1...Done\n");
        for (int i = 0; i < relu_repeats; i++) {
            comm_channel->submit_kernel(dev_relu_code, dev_relu_args);
        }
        if (inidividual_sync) comm_channel->synchronize();
        auto t_k2 = timer::now();
        printf("kernel2...Done\n");
        comm_channel->submit_kernel(dev_linear_code, dev_l2_args);
        if (inidividual_sync) comm_channel->synchronize();
        auto t_k3 = timer::now();
        printf("kernel3...Done\n");
        printf("Run kernels...Done\n");

        printf("Copy output to host...\n");
        comm_channel->copy_d2h(host_output.data(), dev_output, host_output.size() * sizeof(float));
        printf("Copy output to host...Done\n");

        auto t2 = timer::now();
        double s = seconds(t2 - t1);
        times.push_back(s);

        times_in.push_back(seconds(t_input - t1));
        times_k1.push_back(seconds(t_k1 - t_input));
        times_k2.push_back(seconds(t_k2 - t_k1));
        times_k3.push_back(seconds(t_k3 - t_k2));
        times_out.push_back(seconds(t2 - t_k3));
    }

    printf("Host shutdown...\n");
    comm_channel->shutdown();
    printf("Host shutdown... Done\n");

    double mean, std;
    std_mean(times.data(), warmup, repeats, mean, std);
    printf("times mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);

    std_mean(times_in.data(), warmup, repeats, mean, std);
    printf("times_in mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);
    std_mean(times_k1.data(), warmup, repeats, mean, std);
    printf("times_k1 mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);
    std_mean(times_k2.data(), warmup, repeats, mean, std);
    printf("times_k2 mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);
    std_mean(times_k3.data(), warmup, repeats, mean, std);
    printf("times_k3 mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);
    std_mean(times_out.data(), warmup, repeats, mean, std);
    printf("times_out mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);

    std::vector<float> output_ref = read_file<float>("output_ref.bin");
    for (size_t i = 0; i < output_ref.size(); i++) {
        if (output_ref[i] != host_output[i]) {
            printf("Output mismatch at index %zu. Expected %lg Actual %lg\n", i, output_ref[i], host_output[i]);
            //if (i == 10) break;
            return;
        }
    }
    printf("Output is correct\n");
}