struct Circle{
  float2 position;
  float radius;
};

extern "C" void wrapperSdfKernel(uchar4 *image, int width, int height, Circle *circles, int n);
