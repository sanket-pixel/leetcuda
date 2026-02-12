#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 2
// --- KERNEL: matrix_copy ---
__global__ void matrix_copy_kernel(const float *A, float *B, int N) {
  unsigned col_idx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned global_idx = row_idx * N + col_idx;
  if (col_idx < N && row_idx < N) {
    B[global_idx] = A[global_idx];
  }
}

int main() {
  unsigned int N = 64;
  std::vector<float> A(N * N, 2.0);
  std::vector<float> B(N * N, 0.0);

  unsigned bytes = sizeof(float) * N * N;
  float *dA, *dB;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim{BLOCKSIZE, BLOCKSIZE, 1};
  dim3 griddim{(N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE,
               1};
  matrix_copy_kernel<<<griddim, blockdim>>>(dA, dB, N);
  cudaDeviceSynchronize();
  cudaMemcpy(B.data(), dB, bytes, cudaMemcpyDeviceToHost);
}
