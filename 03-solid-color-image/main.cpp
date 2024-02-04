#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "solidImageKernel.h"


#define WIDTH 800
#define HEIGHT 640

Color color = {1.0f, 0.0f, 0.0f};


int main() {
  // Create some memory in host
  std::vector<uchar3> image;
  image.reserve(WIDTH * HEIGHT);

  //================================================================================
  // Create device vars
  uchar3 *d_image;

  // Dedicate memory on device
  cudaMalloc((void **) &d_image, sizeof(uchar3) * WIDTH * HEIGHT);

//  // Copy host to device
//  cudaMemcpy(d_image, image.data(), sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

  // Execute kernel
  wrapperSolidImageKernel(d_image, WIDTH, HEIGHT, color);

  // Copy device to host
  cudaMemcpy(image.data(), d_image, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
  //================================================================================

  stbi_write_png(
      "output.png",
      WIDTH,
      HEIGHT,
      3,
      image.data(),
      WIDTH * sizeof(uchar3)
  );
}
