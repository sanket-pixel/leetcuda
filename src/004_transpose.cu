#include "cmath"
#include <cuda.h>
#include <iostream>
#include <vector>
#define R 5
#define C 6
#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float *input, float *output,
                                        int rows, int cols) {
  unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned r = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned global_id = r * cols + c;
  unsigned global_id_T = c * rows + r;
  if (r < rows && c < cols) {
    output[global_id_T] = input[global_id];
  }
}

int main() {
  std::vector<float> input(R * C, 0.0f);
  std::vector<float> output(C * R, 0.0f);
  float *dinput, *doutput;
  int count = 0;
  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      input[count++] = count;
    }
  }
  cudaMalloc(&dinput, sizeof(float) * R * C);
  cudaMalloc(&doutput, sizeof(float) * R * C);
  cudaMemcpy(dinput, input.data(), sizeof(float) * R * C,
             cudaMemcpyHostToDevice);
  dim3 grid_dim((C + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (R + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  matrix_transpose_kernel<<<grid_dim, block_dim>>>(dinput, doutput, R, C);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), doutput, sizeof(float) * R * C,
             cudaMemcpyDeviceToHost);
  return 0;
}
