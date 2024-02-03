struct Circle{
  float2 position;
  float radius;
};

extern "C" void wrapperSdfKernel(uchar3 *image, int width, int height, Circle *circles, int n);
