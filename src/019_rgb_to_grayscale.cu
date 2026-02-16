#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCKSIZE 1024

__global__ void rgb_to_grayscale_kernel(const float *input, float *output,
                                        int width, int height) {
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < height * width) {
    output[idx] = input[idx * 3] * 0.299f + input[idx * 3 + 1] * 0.587f +
                  input[idx * 3 + 2] * 0.114f;
  }
}

int main() {
  unsigned height = 2;
  unsigned width = 2;
  unsigned total_pixels = height * width;
  unsigned bytes = total_pixels * 3 * sizeof(float);
  std::vector<float> input{255.0, 0.0, 0.0,   0.0,   255.0, 0.0,
                           0.0,   0.0, 255.0, 128.0, 128.0, 128.0};
  std::vector<float> output(total_pixels, 0.0f);
  float *dinput, *doutput;
  cudaMalloc(&dinput, bytes);
  cudaMalloc(&doutput, bytes / 3);
  cudaMemcpy(dinput, input.data(), bytes, cudaMemcpyHostToDevice);

  unsigned grids = ((height * width) + BLOCKSIZE - 1) / BLOCKSIZE;
  rgb_to_grayscale_kernel<<<grids, BLOCKSIZE>>>(dinput, doutput, width, height);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, bytes / 3, cudaMemcpyDeviceToHost);
  for (const auto p : output) {
    std::cout << p << " ";
  }
  std::cout << std::endl;
}
