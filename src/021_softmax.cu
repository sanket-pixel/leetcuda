#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCKSIZE 1024

__device__ void merge_ab(float &ma, float &da, float mb, float db) {
  float new_max = fmaxf(ma, mb);
  da = da * exp(ma - new_max) + db * (mb - new_max);
  ma = new_max;
}
__global__ void online_softmax(const float *input, float *output, int N) {
  __shared__ float sm[32];
  __shared__ float sd[32];
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  float local_m = -INFINITY;
  float local_d = 0.0f;
  for (unsigned i = idx; i < N; i += blockDim.x) {
    float current_value = input[i];
    if (current_value > local_m) {
      local_d = local_d * exp(local_m - current_value) + 1.0f;
      local_m = current_value;
    } else {
      local_d = local_d + exp(current_value - local_m);
    }
  }
  __syncthreads();
  for (unsigned stride = 16; stride >= 1; stride = stride / 2) {
    float other_m = __shfl_down_sync(0xffffffff, local_m, stride);
    float other_d = __shfl_down_sync(0xffffffff, local_d, stride);
    merge_ab(local_m, local_d, other_m, other_d);
  }
}
int main() {
  int N = 4096;
  std::vector<float> input(N, 1.0f);
  unsigned bytes = N * sizeof(float);
  std::vector<float> soft(N, 0.0f);
  float *dinput, *dsoft;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&dinput, sizeof(float) * N);
  cudaMalloc(&dsoft, sizeof(float) * N);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);
  online_softmax<<<1, BLOCKSIZE>>>(dinput, dsoft, N);
  cudaMemcpy(soft.data(), dsoft, bytes / 4, cudaMemcpyDeviceToHost);
}
