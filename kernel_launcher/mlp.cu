#include "mlp.h"

__device__ void linear(
    void* va
) {
    LinearArgs* a = (LinearArgs*)va;
    // input [batch, in]
    // output [batch, out]
    // weight [out, in]
    // bias [out]
    // output = input @ weight.T + bias
    size_t in_features = a->in_features;
    size_t out_features = a->out_features;
    const float* weight = a->weight;
    const float* bias = a->bias;
    size_t batch = a->batch;
    const float* input = a->input;
    float* output = a->output;
    
    for (size_t b = blockIdx.x; b < batch; b += gridDim.x) {
        for (size_t j = threadIdx.x; j < out_features; j += blockDim.x) {
            output[b * out_features + j] = bias[j];
            for (size_t i = 0; i < in_features; i++) {
                output[b * out_features + j] += input[b * in_features + i] * weight[j * in_features + i];
            }
        }
    }
}

__global__ void linear_kernel(LinearArgs* a) {
    linear(a);
}

__device__ void relu(
    void* va
) {
    ReluArgs* a = (ReluArgs*)va;
    size_t size = a->size;
    float* input = a->input;
    float* output = a->output;

    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

__global__ void relu_kernel(ReluArgs* a) {
    relu(a);
}

__device__ void* linear_dont_disappear;
__device__ void* relu_dont_disappear;

__global__ void dummy_kernel() {
    linear_dont_disappear = (void*)linear;
    relu_dont_disappear = (void*)relu;
}


int main(int argc, char** argv) {

    CUDA_MALLOC_MANAGED(float, weight1, sizeof(float) * OUT_FEATURES1 * IN_FEATURES1);

    CUDA_MALLOC_MANAGED(float, bias1, sizeof(float) * OUT_FEATURES1);

    CUDA_MALLOC_MANAGED(float, weight2, sizeof(float) * OUT_FEATURES2 * OUT_FEATURES1);
    CUDA_MALLOC_MANAGED(float, bias2, sizeof(float) * OUT_FEATURES2);
    
    CUDA_MALLOC(float, input, sizeof(float) * BATCH * IN_FEATURES1);
    CUDA_MALLOC(float, internal, sizeof(float) * BATCH * OUT_FEATURES1);
    CUDA_MALLOC(float, output, sizeof(float) * BATCH * OUT_FEATURES2);

    CUDA_MALLOC_MANAGED(LinearArgs, l1_args, sizeof(LinearArgs));
    l1_args->in_features = IN_FEATURES1;
    l1_args->out_features = OUT_FEATURES1;
    l1_args->weight = weight1;
    l1_args->bias = bias1;
    l1_args->batch = BATCH;
    l1_args->input = input;
    l1_args->output = internal;

    CUDA_MALLOC_MANAGED(ReluArgs, relu_args, sizeof(ReluArgs));
    relu_args->size = BATCH * OUT_FEATURES1;
    relu_args->input = internal;
    relu_args->output = internal;

    CUDA_MALLOC_MANAGED(LinearArgs, l2_args, sizeof(LinearArgs));
    l2_args->in_features = OUT_FEATURES1;
    l2_args->out_features = OUT_FEATURES2;
    l2_args->weight = weight2;
    l2_args->bias = bias2;
    l2_args->batch = BATCH;
    l2_args->input = internal;
    l2_args->output = output;

    int warmup = 1;
    int repeats = 3;

    int grid, block;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid, &block, dummy_kernel));
    printf("grid %d block %d\n", grid, block);

    std::vector<float> host_input(BATCH * IN_FEATURES1);
    std::vector<float> host_output(BATCH * OUT_FEATURES2);

    fp_rand(101, weight1, OUT_FEATURES1 * IN_FEATURES1);
    fp_rand(102, bias1, OUT_FEATURES1);
    fp_rand(103, weight2, OUT_FEATURES2 * OUT_FEATURES1);
    fp_rand(104, bias2, OUT_FEATURES2);
    fp_rand(105, host_input.data(), BATCH * IN_FEATURES1);

    std::vector<double> times;
    std::vector<double> times_in, times_out, times_k1, times_k2, times_k3;
    for (int r = 0; r < warmup + repeats; r++) {
        auto t1 = timer::now();
        CUDA_CHECK(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaStreamSynchronize(0));
        auto t_input = timer::now();
        linear_kernel<<<grid, block>>>(l1_args);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        auto t_k1 = timer::now();
        relu_kernel<<<grid, block>>>(relu_args);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        auto t_k2 = timer::now();
        linear_kernel<<<grid, block>>>(l2_args);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        auto t_k3 = timer::now();
        CUDA_CHECK(cudaMemcpy(host_output.data(), output, host_output.size() * sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaStreamSynchronize(0));
        auto t2 = timer::now();
        double s = seconds(t2 - t1);
        times.push_back(s);
        times_in.push_back(seconds(t_input - t1));
        times_k1.push_back(seconds(t_k1 - t_input));
        times_k2.push_back(seconds(t_k2 - t_k1));
        times_k3.push_back(seconds(t_k3 - t_k2));
        times_out.push_back(seconds(t2 - t_k3));
    }
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

    write_file("output_ref.bin", host_output.data(), host_output.size() * sizeof(float));
}