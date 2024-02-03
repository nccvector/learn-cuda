#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "sdfKernel.h"

#define WIDTH 1000
#define HEIGHT 1000
#define N 30


float randf()
{
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
  return dis(e);
}

int main() {
  // Create some spheres on host
  std::vector<Circle> circles;
  for (int i=0; i<N; i++)
  {
    circles.push_back(
        Circle{
          float2{randf(), randf()},
          std::min(std::max(randf(), 0.005f), 0.05f)
        }
    );
  }

  // Create spheres on device
  Circle *dCircles;
  cudaMalloc((void **) &dCircles, sizeof(Circle) * circles.size());
  cudaMemcpy(dCircles, circles.data(), sizeof(Circle) * circles.size(), cudaMemcpyHostToDevice);

  // Create some memory in host
  std::vector<uchar3> image;
  image.reserve(WIDTH * HEIGHT);

  //================================================================================
  // Create device vars
  uchar3 *d_image;

  // Dedicate memory on device
  cudaMalloc((void **) &d_image, sizeof(uchar3) * WIDTH * HEIGHT);

  // Copy host to device
  cudaMemcpy(d_image, image.data(), sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

  // Execute kernel
  wrapperSdfKernel(d_image, WIDTH, HEIGHT, dCircles, circles.size());

  // Copy device to host
  cudaMemcpy(image.data(), d_image, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
  //================================================================================

//  // Debug
//  for(int y=0; y<HEIGHT; y++){
//    for(int x=0; x<WIDTH; x++){
//      std::cout << (int)image[y * WIDTH + x].x << ", ";
//      std::cout << (int)image[y * WIDTH + x].y << ", ";
//      std::cout << (int)image[y * WIDTH + x].z << "\n";
//    }
//  }

  stbi_write_png(
      "output.png",
      WIDTH,
      HEIGHT,
      3,
      image.data(),
      WIDTH * sizeof(uchar3)
  );
}
