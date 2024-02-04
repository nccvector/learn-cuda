#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "sdfKernel.h"

#define WIDTH 1000
#define HEIGHT 1000
#define N 30


float randf() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
  return dis(e);
}

void loadFile(std::string filePath, std::string &out) {
  std::ifstream t(filePath);
  out = std::string(
      (std::istreambuf_iterator<char>(t)),
      std::istreambuf_iterator<char>()
  );
}

void frameBufferSizeCallback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void cudaRenderImageToDeviceBuffer(uchar4 *&d_image) {
  // Create some spheres on host
  std::vector<Circle> circles;
  for (int i = 0; i < N; i++) {
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

  //================================================================================
  // Execute kernel
  wrapperSdfKernel(d_image, WIDTH, HEIGHT, dCircles, circles.size());
}


int main() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);

  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // Load shader and compile shaders
  std::string vertexShaderSource;
  loadFile("shader.vs", vertexShaderSource);
  const char *ccVertexShaderSource = vertexShaderSource.c_str();
  unsigned int vertexShader;
  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &ccVertexShaderSource, NULL);
  glCompileShader(vertexShader);

  std::string fragmentShaderSource;
  loadFile("shader.fs", fragmentShaderSource);
  const char *ccFragmentShaderSource = fragmentShaderSource.c_str();
  unsigned int fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &ccFragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  // Create shader program
  unsigned int shaderProgram;
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  // Creating quad
  float vertices[] = {
      // positions          // colors           // texture coords
      1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
      1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
      -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top left
  };
  unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
  };

  // Create opengl buffers for quad
  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *) 0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *) (3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *) (6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  // Create Texture
  GLuint glTexture;
  glGenTextures(1, &glTexture);
  glBindTexture(GL_TEXTURE_2D, glTexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  std::vector<uchar4> dummy = std::vector<uchar4>(WIDTH * HEIGHT, {255, 0, 255, 255});
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, dummy.data());

  glBindTexture(GL_TEXTURE_2D, 0);

  // CREATE A PBO
  // create pixel buffer object for display
  cudaGraphicsResource_t cudaPboResource;
  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4),
               0, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(
      &cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);

  bool debugFirstFrame = false;

  while (!glfwWindowShouldClose(window)) {
    // Create device image buffer
    uchar4 *deviceImage;

    // Map the deviceImage to PBO
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer(
        (void **) &deviceImage,
        &numBytes,
        cudaPboResource
    );

    // Clear the image memory
    cudaMemset(deviceImage, 0, sizeof(uchar4) * WIDTH * HEIGHT);

    // Render
    cudaRenderImageToDeviceBuffer(deviceImage);

    // Unmap resource
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    cudaStreamSynchronize(0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA,
                    GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glTexture);

    // render container
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(glTexture, "outTexture"), 0);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);   // unbind

    glfwSwapBuffers(window);
    glfwPollEvents();

    std::cout << "IN LOOP\n";
  }
}
