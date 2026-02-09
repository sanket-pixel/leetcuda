#include "cmath"
#include <cuda.h>
#include <iostream>
#define N 1000
#define BLOCK_SIZE 256

__global__ void vecadd(int *da, int *db, int *dc, int n) {
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    dc[id] = da[id] + db[id];
  }
}

int main() {
  int *ha, *hb, *hc;
  int *da, *db, *dc;
  ha = static_cast<int *>(malloc(N * sizeof(int)));
  hb = static_cast<int *>(malloc(N * sizeof(int)));
  hc = static_cast<int *>(malloc(N * sizeof(int)));

  for (int i = 0; i < N; i++) {
    ha[i] = i;
    hb[i] = i + 1;
    hc[i] = 0;
  }
  cudaMalloc(&da, N * sizeof(int));
  cudaMalloc(&db, N * sizeof(int));
  cudaMalloc(&dc, N * sizeof(int));

  cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, N * sizeof(int), cudaMemcpyHostToDevice);

  unsigned block_dim = BLOCK_SIZE;
  unsigned grid_dim = (N + block_dim - 1) / block_dim;
  vecadd<<<grid_dim, block_dim>>>(da, db, dc, N);
  cudaDeviceSynchronize();

  cudaMemcpy(hc, dc, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    std::cout << hc[i] << ", ";
  }
  std::cout << std::endl;
  return 0;
}