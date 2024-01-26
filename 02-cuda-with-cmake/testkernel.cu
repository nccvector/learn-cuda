#include "testkernel.h"

// KERNEL
__global__ void testKernel(float *out, float *a, float *b, int n){
  for(int i=0; i<n; i++)
  {
    out[i] = a[i] + b[i];
  }
}

// CALLER
void kernelCaller(float *out, float *a, float *b, int n){

  float *da, *db, *dout;

  // Dedicate memory on device
  cudaMalloc((void **) &da, sizeof(float) * n);
  cudaMalloc((void **) &db, sizeof(float) * n);
  cudaMalloc((void **) &dout, sizeof(float) * n);

  // Copy host to device
  cudaMemcpy(da, a, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dout, out, sizeof(float) * n, cudaMemcpyHostToDevice);

  // Execute kernel
  testKernel<<<1, 1>>>(dout, da, db, n);

  // Copy device to host
  cudaMemcpy(a, da, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, db, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(out, dout, sizeof(float) * n, cudaMemcpyDeviceToHost);
}
