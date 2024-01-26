#include <iostream>
#include <vector>
#include "testkernel.cuh"

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

  kernelCaller(out.data(), a.data(), b.data(), N);

  for (int i=0; i<N; i++)
  {
    std::cout << out[i] << std::endl;
  }
}