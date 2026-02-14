#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
// --- KERNEL: sigmoid ---
__global__ void swiglu_kernel(const float *input, float *output, int halfN) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < halfN) {
    float silu = (1 / (1 + exp(-1 * input[idx]))) * input[idx];
    output[idx] = silu * input[idx + halfN];
  }
}

int main() {
  int N = 4;
  unsigned bytes = N * sizeof(float);
  std::vector<float> input{1.0, 2.0, 3.0, 4.0};
  std::vector<float> output(N / 2, 0);
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, bytes / 2);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N / 2 + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

  swiglu_kernel<<<griddim, blockdim>>>(dinput, doutput, N / 2);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes / 2, cudaMemcpyDeviceToHost);
  for (const auto &o : output) {
    std::cout << o << " ";
  }
  std::cout << std::endl;
}
