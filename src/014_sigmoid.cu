#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
// --- KERNEL: sigmoid ---
__global__ void silu_kernel(const float *input, float *output, int N) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    output[idx] = (1 / (1 + exp(-1 * input[idx]))) * input[idx];
  }
}

int main() {
  int N = 5;
  unsigned bytes = N * sizeof(float);
  std::vector<float> input{-1.0, -2.0, -3.0, -4.0, -5.0};
  std::vector<float> output(N, 0);
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, bytes);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

  silu_kernel<<<griddim, blockdim>>>(dinput, doutput, N);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes, cudaMemcpyDeviceToHost);
  for (const auto &o : output) {
    std::cout << o << " ";
  }
  std::cout << std::endl;
}
