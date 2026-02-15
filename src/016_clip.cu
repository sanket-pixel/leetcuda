#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCKSIZE 1024

__global__ void clip_kernel(const float *input, float *output, float lo,
                            float hi, int N) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    if (input[idx] < lo) {
      output[idx] = lo;
    } else if (input[idx] > hi) {
      output[idx] = hi;
    } else {
      output[idx] = input[idx];
    }
  }
}

int main() {
  int N = 4;
  unsigned bytes = N * sizeof(float);
  std::vector<float> input{1.5, -2.0, 3.0, 4.5};
  std::vector<float> output(N, 0);
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, bytes);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);
  float lo = 0.0;
  float hi = 3.5;
  clip_kernel<<<griddim, blockdim>>>(dinput, doutput, lo, hi, N);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes, cudaMemcpyDeviceToHost);
  for (const auto &o : output) {
    std::cout << o << " ";
  }
  std::cout << std::endl;
}
