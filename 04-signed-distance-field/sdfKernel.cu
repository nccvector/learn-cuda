#include "sdfKernel.h"

// KERNEL
__global__ void sdfKernel(uchar3 *image, int width, int height, Circle *circles, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  int imageBufferIdx = j * width + i;

  float px = (float) blockIdx.x;
  float py = (float) threadIdx.x;
  float u = (float) px / (width-1);
  float v = (float) py / (height-1);

  float minDist = 1.0f;
  float signedDist = 0.0f;
  for (int cIdx = 0; cIdx < n; cIdx++) {
    float dx = u - circles[cIdx].position.x;
    float dy = v - circles[cIdx].position.y;
    float distance = pow(pow(dx, 2) + pow(dy, 2), 0.5f);

    if (distance < minDist){
      minDist = distance;
      signedDist = minDist - circles[cIdx].radius;
    }
  }

  signedDist = min(signedDist, 1.0f);
  signedDist = max(signedDist, -1.0f);

  float r = max(signedDist, 0.25f);
  float g = max(-1 * signedDist, 0.1f);
  float b = max(1 - pow(abs(signedDist), 0.1f), 0.0f);

  image[imageBufferIdx] = {
      (unsigned char) (r * 255),
      (unsigned char) (g * 255),
      (unsigned char) (b * 255),
  };

//  image[imageBufferIdx] = {
//      (unsigned char) (u * 255),
//      (unsigned char) (v * 255),
//      (unsigned char) (0),
//  };
}

void wrapperSdfKernel(uchar3 *image, int width, int height, Circle *circles, int n) {
  sdfKernel<<<width, height>>>(image, width, height, circles, n);
}