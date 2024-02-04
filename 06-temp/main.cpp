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

void cudaRenderImageToDeviceBuffer(cudaSurfaceObject_t &d_image) {
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
  // Create device vars
  // Dedicate memory on device
  cudaMalloc((void **) &d_image, sizeof(uchar4) * WIDTH * HEIGHT);

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
  unsigned int glTexture;
  glGenTextures(1, &glTexture);
  glBindTexture(GL_TEXTURE_2D, glTexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  glBindTexture(GL_TEXTURE_2D, 0);

  // Create cuda graphics resource
  cudaGraphicsResource *cudaGraphicsResource;
  cudaGraphicsGLRegisterImage(
      &cudaGraphicsResource,
      glTexture,
      GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsWriteDiscard
  );

//  // Create cuda render buffer
//  uchar4 *cudaDeviceRenderBuffer;
//  cudaMalloc((void **) &cudaDeviceRenderBuffer, sizeof(uchar4) * WIDTH * HEIGHT);

  // Create host image
  std::vector<uchar4> hostImage;
  hostImage.reserve(WIDTH * HEIGHT);

  bool debugFirstFrame = false;

  while (!glfwWindowShouldClose(window)) {
    // Map device ptr to the cudaGraphicsResource
    cudaGraphicsMapResources(1, &cudaGraphicsResource, 0);

    cudaArray_t viewCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cudaGraphicsResource, 0, 0);
    cudaResourceDesc cudaResourceDesc;
    cudaResourceDesc.resType = cudaResourceTypeArray;
    cudaResourceDesc.res.array.array = viewCudaArray;

    cudaSurfaceObject_t viewCudaSurfaceObject;
    cudaCreateSurfaceObject(&viewCudaSurfaceObject, &cudaResourceDesc);

    // Render cuda image
    cudaRenderImageToDeviceBuffer(viewCudaSurfaceObject);

    cudaDestroySurfaceObject(viewCudaSurfaceObject);

    // Unmap cuda graphics resources
    cudaGraphicsUnmapResources(1, &cudaGraphicsResource, 0);

    glClearColor(0.2f, 0.2f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

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
