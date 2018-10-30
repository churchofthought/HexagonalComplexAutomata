#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <builtin_types.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuComplex.h>

#include "scatterlife.h" 


// CUDA runtime
//#include <cooperative_groups.h>
//using namespace cooperative_groups;


GLuint rasterTexture;

GLFWwindow* window;


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}









//=========================================
// BEGIN CUDA KERNELS
//=========================================






__device__ Universe univ = {};
__device__ Universe univ2 = {};


__device__ UniImg raster = {};

//__device__ volatile unsigned int gTime = 1; 

//__device__ volatile unsigned int maxParticleCount = 0;


Universe host_univ = {};
UniImg host_raster = {};





// hexagonal particle storage
// 0 N
// 1 NE
// 2 SE
// 3 S
// 4 SW
// 5 NW

__global__ void runAutomata(bool direction){

  Universe* origin;
  Universe* target;
  if (direction){
    origin = &univ;
    target = &univ2;
  }else{
    origin = &univ2;
    target = &univ;
  }

  unsigned int x = blockIdx.x;
  unsigned int y = blockIdx.y;




  unsigned int xm1 = x >= 1 ? x-1 : (UNIVERSE_WIDTH - 1);
  unsigned int xp1 = x < (UNIVERSE_WIDTH - 1) ? x+1 : 0;

  unsigned int ym1 = y >= 1 ? y-1 : (UNIVERSE_HEIGHT - 1);
  unsigned int yp1 = y < (UNIVERSE_HEIGHT - 1) ? y+1 : 0;

  cuFloatComplex z = (*origin)[x][y];

  cuFloatComplex neighborhood[6] = {
    (*origin)   [x] [ym1],   
    (*origin) [xp1] [ym1],
    (*origin) [xp1]   [y],   
    (*origin)   [x] [yp1],   
    (*origin) [xm1] [yp1], 
    (*origin) [xm1]   [y]
  };


  cuFloatComplex tot = cuCaddf(
                        cuCaddf(
                          cuCaddf(neighborhood[0], neighborhood[1]),
                          cuCaddf(neighborhood[2], neighborhood[3])
                        ),
                        cuCaddf(neighborhood[4], neighborhood[5])
                      );

  cuFloatComplex res = cuCmulf(z, tot);
  res = cuCdivf(res, make_cuFloatComplex(cuCabsf(res), 0));
  (*target)[x][y] = res;
}




__device__ float HueToRGB(float v1, float v2, float vH)
{
  if (vH < 0)
    vH += 1;

  if (vH > 1)
    vH -= 1;

  if ((6 * vH) < 1)
    return (v1 + (v2 - v1) * 6 * vH);

  if ((2 * vH) < 1)
    return v2;

  if ((3 * vH) < 2)
    return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

  return v1;
}

__device__ struct RGB HSLToRGB(struct HSL hsl) {
  struct RGB rgb;

  if (hsl.S == 0)
  {
    rgb.R = rgb.G = rgb.B = hsl.L;
  }
  else
  {
    float v1, v2;

    v2 = (hsl.L < 0.5) ? (hsl.L * (1 + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
    v1 = 2 * hsl.L - v2;

    rgb.R = HueToRGB(v1, v2, hsl.H + (1.0f / 3));
    rgb.G = HueToRGB(v1, v2, hsl.H);
    rgb.B = HueToRGB(v1, v2, hsl.H - (1.0f / 3));
  }

  return rgb;
}

__global__ void rasterizeAutomata(){
  unsigned int x = blockIdx.x;
  unsigned int y = blockIdx.y;

  cuFloatComplex z = univ[x][y];

  raster[x][y] = HSLToRGB({
    cuCabsf(z), 
    1.0f, 
    (1.0f + atan2f(cuCimagf(z), cuCrealf(z)) / (3.141592654f)) / 2.0f
  });
}



//=========================================
// END CUDA KERNELS
//=========================================








#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




// void dump_univ(){
//   for (int i = 0; i < UNIVERSE_WIDTH; ++i){
//     for (int z = 0; z < UNIVERSE_HEIGHT; ++z){
//       printf("%3d", host_raster[i][z]);
//     }
//     printf("\n");
//   }
// }


void initOpenGL(){

  glfwInit();

  const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

  // glfwWindowHint(GLFW_RED_BITS, mode->redBits);
  // glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
  // glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
  // glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

  window = glfwCreateWindow(UNIVERSE_WIDTH/2, UNIVERSE_HEIGHT/2, "ScatterLife", NULL, NULL);
  //window = glfwCreateWindow(mode->width, mode->height, "ScatterLife", NULL, NULL);

  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  glfwMakeContextCurrent(window);

  glewInit();

  // setup raster to texture modes
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &rasterTexture);   // generate a texture handler really reccomanded (mandatory in openGL 3.0)
  glBindTexture(GL_TEXTURE_2D, rasterTexture); // tell openGL that we are using the texture 

  glEnable(GL_TEXTURE_2D);

  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  // glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
                 GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
                 GL_NEAREST);
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

  glMatrixMode(GL_PROJECTION);

  GLdouble matrix[16] = {
    sqrt(3.0), 0, 0, 0,
    sqrt(3.0)/2.0, 3.0/2.0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };
  glLoadMatrixd(matrix);
  //glOrtho(0.0f, UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 0.0f, 0.0f, 1.0f);
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}


DWORD WINAPI render( LPVOID lpParam ) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("  Device name: %s\n", prop.name);

  cudaSetDevice(0);
  initOpenGL();

  float scale = 0.5f;


  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
      // rasterize
      rasterizeAutomata<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1),0,cudaStreamPerThread>>>();

      // copy raster back to host
      cudaMemcpyFromSymbolAsync(host_raster, raster, sizeof(UniImg), 0, cudaMemcpyDeviceToHost, cudaStreamPerThread);

      cudaStreamSynchronize(cudaStreamPerThread);

      //glClear(GL_COLOR_BUFFER_BIT);
      
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, RASTER_WIDTH, RASTER_HEIGHT, 0, GL_RGB, GL_FLOAT, host_raster);
      //glGenerateTextureMipmap(rasterTexture);

      glBegin(GL_TRIANGLE_STRIP);

      glTexCoord2f(1.0f, 1.0f); glVertex2f(-scale, -scale);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(-scale, scale);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(scale, -scale);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(scale, scale);

      glEnd();

      glfwSwapBuffers(window);

      glfwPollEvents();
  }

  exit(0);
}

#define RAND_FLOAT ((float)rand()/(float)(RAND_MAX/1.0f))


int main(int argc, char **argv)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("  Device name: %s\n", prop.name);

  cudaSetDevice(0);
  CreateThread(NULL, 0, render, NULL, 0, NULL);

  //initialize automata
  for (int x = 0; x < UNIVERSE_WIDTH; ++x){
    for (int y = 0; y < UNIVERSE_HEIGHT; ++y){
      host_univ[x][y] = make_cuFloatComplex(RAND_FLOAT, RAND_FLOAT);
    }
  }
  
  //host_univ[UNIVERSE_WIDTH/2][UNIVERSE_HEIGHT/2] = make_cuFloatComplex(1.0, 1.0);

  cudaMemcpyToSymbol(univ, host_univ, sizeof(Universe), 0, cudaMemcpyHostToDevice);

  
  for (;;){
    // for (int i = 1; i--;){
      runAutomata<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1),0,cudaStreamPerThread >>>(true);
      runAutomata<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1),0,cudaStreamPerThread >>>(false);
    // }
    cudaStreamSynchronize(cudaStreamPerThread);
  }
}