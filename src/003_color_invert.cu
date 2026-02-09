#include "cmath"
#include <cuda.h>
#include <iostream>
#include <vector>
#define w 2
#define h 1
#define BLOCK_SIZE 256

__global__ void invert_kernel(unsigned char *image, int width, int height) {
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < width * height) {
    image[id * 4] = 255 - image[id * 4];
    image[id * 4 + 1] = 255 - image[id * 4 + 1];
    image[id * 4 + 2] = 255 - image[id * 4 + 2];
  }
}
int main() {
  unsigned char image[8];
  for (int i = 0; i < w * h * 4; i++) {
    image[i] = i;
  }

  unsigned char *dimage;
  cudaMalloc(&dimage, sizeof(unsigned char) * w * h * 4);
  cudaMemcpy(dimage, &image, sizeof(unsigned char) * w * h * 4,
             cudaMemcpyHostToDevice);

  int grid_dim = (w * h + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int block_dim = BLOCK_SIZE;

  invert_kernel<<<grid_dim, block_dim>>>(dimage, 2, 1);
  cudaMemcpy(&image, dimage, sizeof(unsigned char) * w * h * 4,
             cudaMemcpyDeviceToHost);
}
