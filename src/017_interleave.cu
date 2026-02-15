#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCKSIZE 1024
// --- KERNEL: interleave ---
__global__ void interleave_kernel(const float *A, const float *B, float *output,
                                  int N) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    output[2 * idx] = A[idx];
    output[2 * idx + 1] = B[idx];
  }
}

int main() {
  int N = 3;
  unsigned bytes = N * sizeof(float);
  std::vector<float> A{1.0, 2.0, 3.0};
  std::vector<float> B{4.0, 5.0, 6.0};
  std::vector<float> output(2 * N, 0);
  float *dA, *dB, *doutput;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&doutput, bytes * 2);
  cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);
  interleave_kernel<<<griddim, blockdim>>>(dA, dB, doutput, N);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes * 2, cudaMemcpyDeviceToHost);
  for (const auto &o : output) {
    std::cout << o << " ";
  }
  std::cout << std::endl;
}
