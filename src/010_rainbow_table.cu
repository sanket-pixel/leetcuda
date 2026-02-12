#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
__device__ unsigned int fnv1a_hash(int input) {
  const unsigned int FNV_PRIME = 16777619;
  const unsigned int OFFSET_BASIS = 2166136261;

  unsigned int hash = OFFSET_BASIS;

  for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
    unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
    hash = (hash ^ byte) * FNV_PRIME;
  }

  return hash;
}

// --- KERNEL: rainbow_table ---
__global__ void fnv1a_hash_kernel(const int *input, unsigned int *output, int N,
                                  int R) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx];
    for (int i = 0; i < R; i++) {
      output[idx] = fnv1a_hash(output[idx]);
    }
  }
}

int main() {
  int N = 3;
  int R = 2;
  unsigned bytes = sizeof(int) * N;
  std::vector<int> hinput{123, 456, 789};
  int *dinput;
  cudaMalloc(&dinput, bytes);
  cudaMemcpy(dinput, hinput.data(), bytes, cudaMemcpyHostToDevice);

  std::vector<int> hout(N, 0);
  unsigned int *dout;
  cudaMalloc(&dout, bytes);

  unsigned total_threads = N;
  dim3 grid_dim((total_threads + BLOCKSIZE - 1) / BLOCKSIZE);
  dim3 block_dim(BLOCKSIZE, 1, 1);
  fnv1a_hash_kernel<<<grid_dim, block_dim>>>(dinput, dout, N, R);
  cudaDeviceSynchronize();
  cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyHostToDevice);
  for (int i = 0; i < N; i++) {
    std::cout << hout[i] << " ";
  }
  std::cout << std::endl;
}
