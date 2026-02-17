#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCKSIZE 1024

__global__ void reduce(const float *input, float *output, int N) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ float s[BLOCKSIZE];
  if (idx < N) {
    s[threadIdx.x] = input[idx];
  } else {
    s[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride >= 1; stride = stride / 2) {
    if (threadIdx.x < stride) {
      s[threadIdx.x] += s[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, s[0]);
  }
}

int main() {
  int N = 8;
  std::vector<float> input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  unsigned bytes = N * sizeof(float);
  std::vector<float> sum{0};
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, sizeof(float));
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, 1, 1);
  dim3 griddim((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

  reduce<<<griddim, blockdim>>>(dinput, doutput, N);
  cudaDeviceSynchronize();
  cudaMemcpy(sum.data(), doutput, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << sum[0] << std::endl;
}
