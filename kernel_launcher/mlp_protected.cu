#include "mlp.h"

#define SECRET 0xBA0BAB

#define MAX_MESSAGE_DATA 3
#define MAX_MESSAGE_QUEUE 16

struct Message {
    volatile uint64_t data[MAX_MESSAGE_DATA];
};

#define MESSAGE_EXIT 0
#define MESSAGE_RUN_KERNEL 1
#define MESSAGE_COPY_DATA 2

__device__ unsigned int grid_barrier;

struct CommChannel {
    int device_ready = 0;

    volatile int queue_first = 0;
    volatile int queue_last = 0;
    volatile Message message_queue[MAX_MESSAGE_QUEUE];

    cudaStream_t stream;
    CommChannel() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    void submit_kernel(void* func, void* args) volatile {
        while ((queue_last + 1) % MAX_MESSAGE_QUEUE == queue_first) {
            // device queue is full
        }

        message_queue[queue_last].data[0] = MESSAGE_RUN_KERNEL;
        message_queue[queue_last].data[1] = (uint64_t)func;
        message_queue[queue_last].data[2] = (uint64_t)args;

        //printf("data ptr (host) %p\n", message_queue[queue_last].data);

        __sync_synchronize();
        queue_last = (queue_last + 1) % MAX_MESSAGE_QUEUE;
    }

    void copy(void* dst, void* src, size_t size, bool to_device) volatile {
        while ((queue_last + 1) % MAX_MESSAGE_QUEUE == queue_first) {
            // device queue is full
        }

        if (to_device) {
            // encrypt src

        }

        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!to_device) {
            // decrypt dst

        }
    }

    void shutdown() volatile {
        synchronize();
        message_queue[queue_last].data[0] = MESSAGE_EXIT;
        __sync_synchronize();
        queue_last = (queue_last + 1) % MAX_MESSAGE_QUEUE;
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    void synchronize() volatile {
        __sync_synchronize();
        while (queue_last != queue_first) {
            // wait until queue is not empty
        }
    }
};

__global__ void kernel_launcher(volatile CommChannel* comm_channel) {
    // assume here we finished running checksum function
    // and established secret key

    cooperative_groups::details::grid::sync(&grid_barrier);
    if (threadIdx.x == 0 && blockIdx.x == 0) comm_channel->device_ready = 1;
    //__threadfence_system();

    while (1) {
        //cooperative_groups::details::grid::sync(&grid_barrier);
        //if (threadIdx.x == 0 && blockIdx.x == 0) printf("Device waits messages...\n");
        __threadfence_system();
        while (comm_channel->queue_last == comm_channel->queue_first) {}
        //__threadfence_system();
        //cooperative_groups::details::grid::sync(&grid_barrier);
        //if (threadIdx.x == 0 && blockIdx.x == 0) printf("Device waits messages... New message!\n");

        volatile Message* m = &comm_channel->message_queue[comm_channel->queue_first];

        // TODO: decrypt message

        if (m->data[0] == MESSAGE_EXIT) {
            if (threadIdx.x == 0 && blockIdx.x == 0) printf("Exit message\n");
            // exit
            return;
        } else if (m->data[0] == MESSAGE_RUN_KERNEL) {
            //cooperative_groups::details::grid::sync(&grid_barrier);
            //if (threadIdx.x == 0 && blockIdx.x == 0) printf("Run kernel message\n");

            //printf("data ptr (device) %p\n", m->data);
            // run kernel
            //uint64_t kernel_pointer = (uint64_t)fooptr;
            uint64_t kernel_pointer = (uint64_t)m->data[1];
            uint64_t args_pointer = m->data[2];

            SAGEKernelPtr kernel = (SAGEKernelPtr)(void*)kernel_pointer;

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("kernel_pointer[0] 0x%" PRIx64 "\n", ((uint64_t*)kernel_pointer)[0]);
            //     printf("kernel_pointer[1] 0x%" PRIx64 "\n", ((uint64_t*)kernel_pointer)[1]);
            //     printf("kernel_pointer[2] 0x%" PRIx64 "\n", ((uint64_t*)kernel_pointer)[2]);
            //     printf("kernel_pointer[3] 0x%" PRIx64 "\n", ((uint64_t*)kernel_pointer)[3]);
            // }

            //printf("kernel_pointer %" PRIu64 " args_pointer %" PRIu64 "\n", kernel_pointer, args_pointer);
            kernel((void*)args_pointer);
            // cooperative_groups::details::grid::sync(&grid_barrier);
            // if (threadIdx.x == 0 && blockIdx.x == 0) printf("Kernel finished\n");
        } else if (m->data[0] == MESSAGE_COPY_DATA) {
            // copy data
        } else {
            if (threadIdx.x == 0 && blockIdx.x == 0) printf("Unknown message\n");
        }

        //printf("Before barrier\n");
        cooperative_groups::details::grid::sync(&grid_barrier);
        //printf("After barrier\n");
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            comm_channel->queue_first = (comm_channel->queue_first + 1) % MAX_MESSAGE_QUEUE;
        }
        cooperative_groups::details::grid::sync(&grid_barrier);
    }
}

int main() {
    // assume this code runs inside SGX

    std::ifstream linear_in("linear.bin", std::ios::binary | std::ios::ate);
    size_t linear_code_size = linear_in.tellg();
    printf("Linear code size %zu\n", linear_code_size);
    linear_in.seekg(0, std::ios::beg);
    std::vector<uint8_t> linear_code(linear_code_size);
    linear_in.read((char*)linear_code.data(), linear_code_size);

    std::ifstream relu_in("relu.bin", std::ios::binary | std::ios::ate);
    size_t relu_code_size = relu_in.tellg();
    printf("Relu code size %zu\n", relu_code_size);
    relu_in.seekg(0, std::ios::beg);
    std::vector<uint64_t> relu_code(relu_code_size);
    relu_in.read((char*)relu_code.data(), relu_code_size);

    std::vector<float> host_weight1(OUT_FEATURES1 * IN_FEATURES1);
    std::vector<float> host_bias1(OUT_FEATURES1);
    std::vector<float> host_weight2(OUT_FEATURES2 * OUT_FEATURES1);
    std::vector<float> host_bias2(OUT_FEATURES2);

    std::vector<float> host_input(BATCH * IN_FEATURES1);
    std::vector<float> host_internal(BATCH * OUT_FEATURES1);
    std::vector<float> host_output(BATCH * OUT_FEATURES2);

    CUDA_MALLOC(uint8_t, dev_linear_code, linear_code.size());
    CUDA_MALLOC(uint8_t, dev_relu_code, relu_code.size());

    CUDA_MALLOC(float, dev_weight1, sizeof(float) * OUT_FEATURES1 * IN_FEATURES1);
    CUDA_MALLOC(float, dev_bias1, sizeof(float) * OUT_FEATURES1);
    CUDA_MALLOC(float, dev_weight2, sizeof(float) * OUT_FEATURES2 * OUT_FEATURES1);
    CUDA_MALLOC(float, dev_bias2, sizeof(float) * OUT_FEATURES2);

    CUDA_MALLOC(float, dev_input, sizeof(float) * BATCH * IN_FEATURES1);
    CUDA_MALLOC(float, dev_internal, sizeof(float) * BATCH * OUT_FEATURES1);
    CUDA_MALLOC(float, dev_output, sizeof(float) * BATCH * OUT_FEATURES2);

    // create communication channel with device
    CommChannel* comm_channel;
    CUDA_CHECK(cudaMallocHost(&comm_channel, sizeof(CommChannel)));
    new (comm_channel) CommChannel();
    volatile CommChannel* vcomm_channel = comm_channel;

    int grid, block;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid, &block, kernel_launcher));
    printf("grid %d block %d\n", grid, block);

    // run checksum verification and key establishment
    kernel_launcher<<<grid, block>>>(comm_channel);
    CUDA_CHECK(cudaPeekAtLastError());

    printf("waiting for device...\n");
    while (!vcomm_channel->device_ready) {}
    printf("waiting for device... Done\n");

    // TODO: encrypt model and input

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy encrypted model to device
    CUDA_CHECK(cudaMemcpyAsync(dev_weight1, &host_weight1[0], host_weight1.size() * sizeof(float), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_bias1, &host_bias1[0], host_bias1.size() * sizeof(float), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_weight2, &host_weight2[0], host_weight2.size() * sizeof(float), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_bias2, &host_bias2[0], host_bias2.size() * sizeof(float), cudaMemcpyDefault, stream));



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

    LinearArgs* dev_l1_args;
    CUDA_CHECK(cudaMallocAsync(&dev_l1_args, sizeof(LinearArgs), stream));
    ReluArgs* dev_relu_args;
    CUDA_CHECK(cudaMallocAsync(&dev_relu_args, sizeof(ReluArgs), stream));
    LinearArgs* dev_l2_args;
    CUDA_CHECK(cudaMallocAsync(&dev_l2_args, sizeof(LinearArgs), stream));

    CUDA_CHECK(cudaMemcpyAsync(dev_l1_args, &l1_args, sizeof(LinearArgs), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_relu_args, &relu_args, sizeof(ReluArgs), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_l2_args, &l2_args, sizeof(LinearArgs), cudaMemcpyDefault, stream));

    CUDA_CHECK(cudaMemcpyAsync(dev_linear_code, linear_code.data(), linear_code.size(), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_relu_code, relu_code.data(), relu_code.size(), cudaMemcpyDefault, stream));

    vcomm_channel->copy(dev_linear_code, linear_code.data(), linear_code.size(), 1);
    vcomm_channel->copy(dev_relu_code, relu_code.data(), relu_code.size(), 1);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("Start benchmarking...\n");

    std::vector<double> times;
    for (int r = 0; r < warmup + repeats; r++) {
        auto t1 = timer::now();

        //printf("Host memcpy async input...\n");
        vcomm_channel->copy(dev_input, host_input.data(), host_input.size() * sizeof(float), true);
        //printf("Host memcpy async input... Done\n");

        //printf("Submitting kernel %" PRIx64 "(%" PRIx64 ")...\n", dev_linear_code, dev_l1_args);
        vcomm_channel->submit_kernel(dev_linear_code, dev_l1_args);
        //printf("Submitting kernel...Done\n");

        vcomm_channel->submit_kernel(dev_relu_code, dev_relu_args);
        vcomm_channel->submit_kernel(dev_linear_code, dev_l2_args);
        //printf("Synchronizing...\n");
        vcomm_channel->synchronize();
        //printf("Synchronizing... Done\n");

        //printf("Host memcpy async output...\n");
        vcomm_channel->copy(host_output.data(), dev_output, host_output.size() * sizeof(float), false);
        //printf("Host memcpy async output... Done\n");

        auto t2 = timer::now();
        double s = seconds(t2 - t1);
        times.push_back(s);
    }

    printf("Host shutdown...\n");
    vcomm_channel->shutdown();
    printf("Host shutdown... Done\n");

    double mean, std;
    std_mean(times.data(), warmup, repeats, mean, std);
    printf("mean %.2f ms std %.2f ms\n", mean * 1e3, std * 1e3);
}