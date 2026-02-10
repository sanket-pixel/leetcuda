#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
// --- KERNEL: reverse_array ---
__global__ void reverse_array_kernel(float *input, int N) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N / 2) {
    float temp = input[idx];
    input[idx] = input[N - idx - 1];
    input[N - idx - 1] = temp;
  }
}

int main() {
  int N = 8;
  unsigned bytes = sizeof(float) * N;
  std::vector<float> hinput(N, 0);
  for (int i = 1; i <= N; i++) {
    hinput[i] = static_cast<float>(i);
  }
  float *dinput;
  cudaMalloc(&dinput, bytes);
  cudaMemcpy(dinput, hinput.data(), bytes, cudaMemcpyHostToDevice);
  unsigned total_threads = N / 2;
  dim3 grid_dim((total_threads + BLOCKSIZE - 1) / BLOCKSIZE);
  dim3 block_dim(BLOCKSIZE, 1, 1);
  reverse_array_kernel<<<grid_dim, block_dim>>>(dinput, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hinput.data(), dinput, bytes, cudaMemcpyHostToDevice);
  for (int i = 0; i < N; i++) {
    std::cout << hinput[i];
  }
  std::cout << std::endl;
}
