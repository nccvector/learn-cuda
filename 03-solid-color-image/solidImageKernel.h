struct Color{
  float r, g, b;
};

extern "C" void wrapperSolidImageKernel(uchar3 *image, int width, int height, Color color);