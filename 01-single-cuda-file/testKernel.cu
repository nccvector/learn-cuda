#include <iostream>
#include <vector>

#define N 1000

// KERNEL
__global__ void testKernel(float *out, float *a, float *b, int n){
  for(int i=0; i<N; i++)
  {
    out[i] = a[i] + b[i];
  }
}

// CALLER
void kernelCaller(float *out, float *a, float *b, int n){

  float *da, *db, *dout;

  // Dedicate memory on device
  cudaMalloc((void **) &da, sizeof(float) * N);
  cudaMalloc((void **) &db, sizeof(float) * N);
  cudaMalloc((void **) &dout, sizeof(float) * N);

  // Copy host to device
  cudaMemcpy(da, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dout, out, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Execute kernel
  testKernel<<<1, 1>>>(dout, da, db, N);

  // Copy device to host
  cudaMemcpy(a, da, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, db, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(out, dout, sizeof(float) * N, cudaMemcpyDeviceToHost);
}

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

  kernelCaller(out.data(), a.data(), b.data(), N);

  for (int i=0; i<N; i++)
  {
    std::cout << out[i] << std::endl;
  }
}
