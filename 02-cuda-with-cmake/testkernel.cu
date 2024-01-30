#include "testkernel.h"

// KERNEL
__global__ void testKernel(float *out, float *a, float *b, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

void wrapperKernel(float *out, float *a, float *b, int n) {
  testKernel<<<1, 1>>>(out, a, b, n);
}