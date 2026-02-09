#include <iostream>
#include <cuda_runtime.h>

// --- KERNEL: sample ---
__global__ void sample_kernel(float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement sample logic
        d_out[idx] = d_in[idx];
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate Unified Memory (Read by CPU & GPU)
    float *data;
    cudaMallocManaged(&data, bytes);

    // Initialize
    for(int i=0; i<N; i++) data[i] = 1.0f;

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    sample_kernel<<<blocks, threads>>>(data, data, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel: sample | Time: %.3f ms\n", ms);

    cudaFree(data);
    return 0;
}
