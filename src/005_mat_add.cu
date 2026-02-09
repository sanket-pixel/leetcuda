#include "cmath"
#include <cuda.h>
#include <iostream>
#include <vector>
#define n 2
#define BLOCK_SIZE 32

__global__ void matrix_add(const float *A, const float *B, float *C, int N) {
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n * n) {
    C[id] = A[id] + B[id];
  }
}

int main() {
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C = {0.0f, 0.0f, 0.0f, 0.0f};

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * n * n);
  cudaMalloc(&dB, sizeof(float) * n * n);
  cudaMalloc(&dC, sizeof(float) * n * n);

  cudaMemcpy(dA, A.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

  int block_dim = BLOCK_SIZE;
  int grid_dim = (n * n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  matrix_add<<<grid_dim, block_dim>>>(dA, dB, dC, n);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
}
