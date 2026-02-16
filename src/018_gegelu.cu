#include <cuda_runtime.h>
#include <iostream>

#include <vector>
#define BLOCKSIZE 1024

__global__ void geglu_kernel(const float *input, float *output, int halfN) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < halfN) {
    output[idx] = input[idx] * (input[idx + halfN] / 2) *
                  (1.0f + erff(input[idx + halfN] / sqrtf(2)));
  }
}

int main() {
  int N = 4;
  unsigned bytes = N * sizeof(float);
  std::vector<float> input{2.0, -1.0, 1.0, 0.5};
  std::vector<float> output(N / 2, 0);
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, bytes / 2);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N / 2 + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

  geglu_kernel<<<griddim, blockdim>>>(dinput, doutput, N / 2);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes / 2, cudaMemcpyDeviceToHost);
  for (const auto &o : output) {
    std::cout << o << " ";
  }
  std::cout << std::endl;
}