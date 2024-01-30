#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "testkernel.h"

#define N 1000


int main() {
  // Create some memory in host
  std::vector<float> a, b, out;
  a.reserve(N);
  b.reserve(N);
  out.reserve(N);

  for (int i = 0; i < N; i++) {
    a[i] = 2;
    b[i] = 3;
    out[i] = 0;
  }

  //================================================================================
  // Create device vars
  float *da, *db, *dout;

  // Dedicate memory on device
  cudaMalloc((void **) &da, sizeof(float) * N);
  cudaMalloc((void **) &db, sizeof(float) * N);
  cudaMalloc((void **) &dout, sizeof(float) * N);

  // Copy host to device
  cudaMemcpy(da, a.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dout, out.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

  // Execute kernel
  wrapperKernel(dout, da, db, N);

  // Copy device to host
  cudaMemcpy(a.data(), da, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), db, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(out.data(), dout, sizeof(float) * N, cudaMemcpyDeviceToHost);
  //================================================================================

  for (int i = 0; i < N; i++) {
    std::cout << out[i] << std::endl;
  }
}
