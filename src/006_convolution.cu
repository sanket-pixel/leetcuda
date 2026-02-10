#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024
// --- KERNEL: convolution ---
__global__ void convolution_kernel(float *input, float *kernel, float *output,
                                   int input_size, int kernel_size) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < input_size - kernel_size + 1) {
    float sum = 0.0f;
    for (int j = 0; j < kernel_size; j++) {
      sum += input[idx + j] * kernel[j];
    }
    output[idx] = sum;
  }
}

int main() {
  int N = 5;
  std::vector<float> ha(N, 0);
  for (int i = 1; i <= N; i++) {
    ha[i] = static_cast<float>(i);
  }
  float *da;
  cudaMalloc(&da, sizeof(float) * N);
  cudaMemcpy(da, ha.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

  std::vector<float> hk{1, 0, -1};
  float *dk;
  cudaMalloc(&dk, sizeof(float) * 3);
  cudaMemcpy(dk, hk.data(), sizeof(float) * 3, cudaMemcpyHostToDevice);

  int out_size = N - hk.size() + 1;
  std::vector<float> hout(out_size, 0);
  float *dout;
  cudaMalloc(&dout, sizeof(float) * out_size);

  dim3 grid_dim((out_size + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);
  dim3 block_dim(BLOCKSIZE, 1, 1);

  convolution_kernel<<<grid_dim, block_dim>>>(da, dk, dout, N, hk.size());
  cudaDeviceSynchronize();
  cudaMemcpy(hout.data(), dout, sizeof(float) * out_size,
             cudaMemcpyHostToDevice);

  for (int i = 0; i < out_size; i++) {
    std::cout << hout[i];
  }
  std::cout << std::endl;
  return 0;
}
