struct Circle{
  float2 position;
  float radius;
};

extern "C" void wrapperSdfKernel(cudaSurfaceObject_t &image, int width, int height, Circle *circles, int n);
