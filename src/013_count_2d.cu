#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#define BLOCKSIZE 16
// --- KERNEL: relu ---
__global__ void count_2d_equal_kernel(const int *input, int *output, int N,
                                      int M, int K) {
  unsigned row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned col_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (row_idx < N && col_idx < M) {
    unsigned global_idx = row_idx * M + col_idx;
    if (input[global_idx] == K) {
      atomicAdd(output, 1);
    }
  }
}

int main() {
  int N = 10;
  int M = 12;
  int K = 1;
  unsigned bytes = N * M * sizeof(int);
  std::mt19937 gen(std::random_device{}());
  std::vector<int> input(N * M, 0);
  for (int i = 0; i < N * M; i++) {
    input[i] = std::uniform_int_distribution<int>(1, 1)(gen);
  }
  int output = 0;
  int *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, sizeof(int));
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);
  dim3 blockdim(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 griddim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE,
               1);
  count_2d_equal_kernel<<<griddim, blockdim>>>(dinput, doutput, N, M, K);
  cudaDeviceSynchronize();
  cudaMemcpy(&output, doutput, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << output << std::endl;
}
