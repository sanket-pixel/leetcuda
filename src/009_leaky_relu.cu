#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCKSIZE 1024
// --- KERNEL: relu ---
__global__ void relu_kernel(const float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    if (input[idx] > 0) {
      output[idx] = input[idx];
    } else {
      output[idx] = 0.01 * input[idx];
    }
  }
}

int main() {
  int N = 4;
  unsigned bytes = sizeof(float) * N;
  std::vector<float> hinput{1.0, -2.0, 3.0, -4.0};
  float *dinput;
  cudaMalloc(&dinput, bytes);
  cudaMemcpy(dinput, hinput.data(), bytes, cudaMemcpyHostToDevice);

  std::vector<float> hout(N, 0);
  float *dout;
  cudaMalloc(&dout, bytes);

  unsigned total_threads = N;
  dim3 grid_dim((total_threads + BLOCKSIZE - 1) / BLOCKSIZE);
  dim3 block_dim(BLOCKSIZE, 1, 1);
  relu_kernel<<<grid_dim, block_dim>>>(dinput, dout, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyHostToDevice);
  for (int i = 0; i < N; i++) {
    std::cout << hout[i] << " ";
  }
  std::cout << std::endl;
}
