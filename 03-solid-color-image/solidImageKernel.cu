#include "solidImageKernel.h"

// KERNEL
__global__ void solidImageKernel(uchar3 *image, int width, int height, Color color) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  image[j * width + i] = {
      (unsigned char)(color.r * 255),
      (unsigned char)(color.g * 255),
      (unsigned char)(color.b * 255),
  };
}

void wrapperSolidImageKernel(uchar3 *image, int width, int height, Color color) {
  solidImageKernel<<<width, height>>>(image, width, height, color);
}