#include "mlp.h"

#define SECRET 0xBA0BAB

#define MAX_MESSAGE_DATA 4
#define MAX_MESSAGE_QUEUE 16

struct Message {
    volatile uint64_t data[MAX_MESSAGE_DATA];
};

#define MESSAGE_EXIT 0
#define MESSAGE_RUN_KERNEL 1
#define MESSAGE_COPY_H2D 2
#define MESSAGE_COPY_D2H 3
#define MESSAGE_ALLOC 4

__device__ unsigned int grid_barrier;

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
        message_queue[queue_last].data[0] = MESSAGE_ALLOC;
        message_queue[queue_last].data[1] = (uint64_t)size;
        commit_message_to_queue();
        synchronize();
        return allocated_mem;
    }

    void submit_kernel(void* func, void* args) {
        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = MESSAGE_RUN_KERNEL;
        message_queue[queue_last].data[1] = (uint64_t)func;
        message_queue[queue_last].data[2] = (uint64_t)args;
        commit_message_to_queue();

        synchronize();
    }

    void copy_h2d(void* dst, void* src, size_t size) {
        // allocate mem on device        
        if (size > tmp_buf_size) tmp_buff_realloc(size);
        CUDA_CHECK(cudaMemcpyAsync(tmp_buf, src, size, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = MESSAGE_COPY_H2D;
        message_queue[queue_last].data[1] = (uint64_t)dst;
        message_queue[queue_last].data[2] = (uint64_t)tmp_buf;
        message_queue[queue_last].data[3] = (uint64_t)size;
        commit_message_to_queue();

        synchronize();
    }

    void copy_d2h(void* dst, void* src, size_t size) {
        // allocate mem on device
        if (size > tmp_buf_size) tmp_buff_realloc(size);

        wait_while_queue_is_full();
        message_queue[queue_last].data[0] = MESSAGE_COPY_D2H;
        message_queue[queue_last].data[1] = (uint64_t)tmp_buf;
        message_queue[queue_last].data[2] = (uint64_t)src;
        message_queue[queue_last].data[3] = (uint64_t)size;
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

__global__ void kernel_launcher(volatile CommChannel* comm_channel) {
    // assume here we finished running checksum function
    // and established secret key

    cooperative_groups::details::grid::sync(&grid_barrier);
    if (threadIdx.x == 0 && blockIdx.x == 0) comm_channel->device_ready = 1;

    while (1) {
        cooperative_groups::details::grid::sync(&grid_barrier);
        __threadfence_system();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (comm_channel->queue_last == comm_channel->queue_first) {}
        }
        cooperative_groups::details::grid::sync(&grid_barrier);

        volatile Message* m = &comm_channel->message_queue[comm_channel->queue_first];

        // TODO: decrypt message

        if (m->data[0] == MESSAGE_EXIT) {
            if (threadIdx.x == 0 && blockIdx.x == 0) printf("Exit message\n");
            // exit
            return;
        } else if (m->data[0] == MESSAGE_RUN_KERNEL) {
            SAGEKernelPtr kernel = (SAGEKernelPtr)m->data[1];
            void* args = (void*)m->data[2];
            kernel(args);
        } else if (m->data[0] == MESSAGE_COPY_H2D) {
            char* dst = (char*) m->data[1];
            char* src = (char*) m->data[2];
            size_t size = m->data[3];
            for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
                dst[i] = src[i];
            }
        } else if (m->data[0] == MESSAGE_COPY_D2H) {
            char* dst = (char*) m->data[1];
            char* src = (char*) m->data[2];
            size_t size = m->data[3];
            for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
                dst[i] = src[i];
            }
        } else if (m->data[0] == MESSAGE_ALLOC) {
            if (threadIdx.x + blockIdx.x == 0) {
                comm_channel->allocated_mem = malloc(m->data[1]);
            }
        } else {
            if (threadIdx.x == 0 && blockIdx.x == 0) printf("Unknown message\n");
        }

        cooperative_groups::details::grid::sync(&grid_barrier);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            comm_channel->queue_first = (comm_channel->queue_first + 1) % MAX_MESSAGE_QUEUE;
        }
        cooperative_groups::details::grid::sync(&grid_barrier);
    }
}

int main() {
    // assume this code runs inside SGX

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

    std::vector<float> host_input(BATCH * IN_FEATURES1);
    std::vector<float> host_internal(BATCH * OUT_FEATURES1);
    std::vector<float> host_output(BATCH * OUT_FEATURES2);

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
    kernel_launcher<<<grid, block>>>(comm_channel);
    CUDA_CHECK(cudaPeekAtLastError());

    printf("waiting for device...\n");
    while (!vcomm_channel->device_ready) {}
    printf("waiting for device... Done\n");

    auto dev_weight1 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES1 * IN_FEATURES1);
    auto dev_bias1 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES1);
    auto dev_weight2 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES2 * OUT_FEATURES1);
    auto dev_bias2 = (float*) comm_channel->alloc(sizeof(float) * OUT_FEATURES2);

    auto dev_input = (float*) comm_channel->alloc(sizeof(float) * BATCH * IN_FEATURES1);
    auto dev_internal = (float*) comm_channel->alloc(sizeof(float) * BATCH * OUT_FEATURES1);
    auto dev_output = (float*) comm_channel->alloc(sizeof(float) * BATCH * OUT_FEATURES2);

    int warmup = 1;
    int repeats = 3;
    // int warmup = 0;
    // int repeats = 1;

    LinearArgs l1_args; 
    l1_args.in_features = IN_FEATURES1;
    l1_args.out_features = OUT_FEATURES1;
    l1_args.weight = dev_weight1;
    l1_args.bias = dev_bias1;
    l1_args.batch = BATCH;
    l1_args.input = dev_input;
    l1_args.output = dev_internal;

    ReluArgs relu_args;
    relu_args.size = BATCH * OUT_FEATURES1;
    relu_args.input = dev_internal;
    relu_args.output = dev_internal;

    LinearArgs l2_args;
    l2_args.in_features = OUT_FEATURES1;
    l2_args.out_features = OUT_FEATURES2;
    l2_args.weight = dev_weight2;
    l2_args.bias = dev_bias2;
    l2_args.batch = BATCH;
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
    fp_rand(105, host_input.data(), BATCH * IN_FEATURES1);


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
        comm_channel->copy_h2d(dev_input, host_input.data(), host_input.size() * sizeof(float));
        printf("Copy input to device...Done\n");
        auto t_input = timer::now();

        printf("Run kernels...\n");
        comm_channel->submit_kernel(dev_linear_code, dev_l1_args);
        auto t_k1 = timer::now();
        printf("kernel1...Done\n");
        comm_channel->submit_kernel(dev_relu_code, dev_relu_args);
        auto t_k2 = timer::now();
        printf("kernel2...Done\n");
        comm_channel->submit_kernel(dev_linear_code, dev_l2_args);
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
    printf("mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);

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