#include "cmath"
#include <cuda.h>
#include <iostream>
#include <vector>
#define M 2
#define N 2
#define K 2
#define BLOCK_SIZE 32

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int m, int n, int k) {
  unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned global_id = k * row + col;
  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += A[n * row + i] * B[i * k + col];
    }
    C[global_id] = sum;
  }
}

int main() {
  std::vector<float> A(M * N, 0.0f);
  std::vector<float> B(N * K, 0.0f);
  std::vector<float> C(M * K, 1.0f);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * N);
  cudaMalloc(&dB, sizeof(float) * N * K);
  cudaMalloc(&dC, sizeof(float) * M * K);

  A = {1.0f, 2.0f, 3.0f, 4.0f};
  B = {5.0f, 6.0f, 7.0f, 8.0f};
  cudaMemcpy(dA, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 grid_dim((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  matrix_multiplication_kernel<<<grid_dim, block_dim>>>(dA, dB, dC, M, N, K);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), dC, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;
}