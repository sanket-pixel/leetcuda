#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
__global__ void count_equal_kernel(const int *input, int *output, int N,
                                   int K) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    if (input[idx] == K) {
      atomicAdd(output, 1);
    }
  }
}

int main() {
  int N = 5;
  int K = 1;
  std::vector<int> input{1, 2, 3, 4, 1};
  unsigned bytes = N * sizeof(int);
  std::vector<int> output{0};
  int *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, sizeof(int));
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

  count_equal_kernel<<<griddim, blockdim>>>(dinput, doutput, N, K);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << output[0] << std::endl;
}
