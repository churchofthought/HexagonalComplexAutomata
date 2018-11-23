#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <builtin_types.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>


#define _USE_MATH_DEFINES // for C  
#include <math.h>  

#include <thrust/complex.h>


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






__device__ UniverseDevice univ = {};
__device__ UniverseDevice univ2 = {};


__device__ UniImg raster = {};

//__device__ volatile unsigned int gTime = 1; 

__device__ volatile float maxVal;


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
    origin = (Universe*)(&univ);
    target = (Universe*)(&univ2);
  }else{
    origin = (Universe*)(&univ2);
    target = (Universe*)(&univ);
  }

  unsigned int x = blockIdx.x;
  unsigned int y = blockIdx.y;




  unsigned int xm1 = x == 0 ? (UNIVERSE_WIDTH - 1) : x-1;
  unsigned int xp1 = x == (UNIVERSE_WIDTH - 1) ? 0 : x+1;

  unsigned int ym1 = y == 0 ? (UNIVERSE_HEIGHT - 1) : y-1;
  unsigned int yp1 = y == (UNIVERSE_HEIGHT - 1) ? 0 : y+1;

  thrust::complex<float> prevZ = (*target)[x][y];
  thrust::complex<float> z = (*origin)[x][y];

  thrust::complex<float> neighborhood[6] = {
    (*origin)   [x] [ym1],   
    (*origin) [xp1] [ym1],
    (*origin) [xp1]   [y],   
    (*origin)   [x] [yp1],   
    (*origin) [xm1] [yp1], 
    (*origin) [xm1]   [y]
  };

  thrust::complex<float> unit_vectors[6] = {
    thrust::complex<float>(0.0f, 1.0f),
    thrust::complex<float>(sqrtf(3.0f)/2.0f, 1.0f/2.0f),
    thrust::complex<float>(sqrtf(3.0f)/2.0f, -1.0f/2.0f),
    thrust::complex<float>(0.0f, -1.0f),
    thrust::complex<float>(-sqrtf(3.0f)/2.0f, -1.0f/2.0f),
    thrust::complex<float>(-sqrtf(3.0f)/2.0f, 1.0f/2.0f)
  };

  thrust::complex<float> res = 0;

  for (int i = 0; i < 5; ++i){
    res += unit_vectors[i] * max(0.0f,
      neighborhood[(i + 3) % 6].real() * unit_vectors[i].real() + 
      neighborhood[(i + 3) % 6].imag() * unit_vectors[i].imag()
    );
  }
  (*target)[x][y] = (1.0f/3.0f) * res + (1.0f/2.0f) * z;

  // (*target)[x][y] = thrust::pow(z - prevZ, 2) + 0.01 *(neighborhood[0] + 
  //                           neighborhood[1] +
  //                           neighborhood[2] +
  //                           neighborhood[3] +
  //                           neighborhood[4] +
  //                           neighborhood[5]
  //                           - 6*z);

  // (*target)[x][y] = (
  //   neighborhood[0] + 
  //   neighborhood[1] +
  //   neighborhood[2] +
  //   neighborhood[3] +
  //   neighborhood[4] +
  //   neighborhood[5]
  // ) / 6.0f;
}




__device__ float HueToRGB(float v1, float v2, float vH)
{
  if (vH < 0.0f)
    vH += 1.0f;

  if (vH > 1.0f)
    vH -= 1.0f;

  if ((6.0f * vH) < 1.0f)
    return (v1 + (v2 - v1) * 6 * vH);

  if ((2.0f * vH) < 1.0f)
    return v2;

  if ((3.0f * vH) < 2.0f)
    return (v1 + (v2 - v1) * ((2.0f / 3.0f) - vH) * 6.0f);

  return v1;
}

__device__ struct RGB HSLToRGB(struct HSL hsl) {
  struct RGB rgb;

  if (hsl.S == 0.0f)
  {
    rgb.R = rgb.G = rgb.B = hsl.L;
  }
  else
  {
    float v1, v2;

    v2 = (hsl.L < 0.5f) ? (hsl.L * (1.0f + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
    v1 = 2.0f * hsl.L - v2;

    rgb.R = HueToRGB(v1, v2, hsl.H + (1.0f / 3.0f));
    rgb.G = HueToRGB(v1, v2, hsl.H);
    rgb.B = HueToRGB(v1, v2, hsl.H - (1.0f / 3.0f));
  }

  return rgb;
}



__global__ void rasterizeAutomata1(){
  atomicMax((unsigned int*) &maxVal, __float_as_uint(
      thrust::abs(
      ((Universe&)univ)[blockIdx.x][blockIdx.y]
    )
  ));
}

__global__ void rasterizeAutomata2(){
  unsigned int x = blockIdx.x;
  unsigned int y = blockIdx.y;

  thrust::complex<float> z = ((Universe&)univ)[x][y];

  raster[x][y] = HSLToRGB({
    0.5f + thrust::arg(z) / float(M_PI_2),
    1.0f, 
    1.0f - powf(thrust::abs(z) / maxVal, 0.5f)
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




void dump_univ(){
  cudaMemcpyFromSymbol(host_univ, univ, sizeof(Universe), 0, cudaMemcpyDeviceToHost);
  for (int i = 0; i < UNIVERSE_WIDTH; ++i){
    for (int z = 0; z < UNIVERSE_HEIGHT; ++z){
      printf("(%.2f, %.2f)", host_univ[i][z].real(), host_univ[i][z].imag());
    }
    printf("\n");
  }
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_D && (mods & GLFW_MOD_ALT) && action == GLFW_RELEASE)
        dump_univ();
}

void initOpenGL(){

  glfwInit();

  const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

  // glfwWindowHint(GLFW_RED_BITS, mode->redBits);
  // glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
  // glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
  // glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

  int width = mode->width*.75;//UNIVERSE_WIDTH;
  int height = mode->height*.75;//UNIVERSE_HEIGHT;
  window = glfwCreateWindow(width, height, "ScatterLife", NULL, NULL);
  glfwSetWindowPos(window, (mode->width - width) / 2, (mode->height - height) / 2);

  glfwSetKeyCallback(window, key_callback);
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
                 GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
                 GL_LINEAR);
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

  glMatrixMode(GL_PROJECTION);

  GLdouble matrix[16] = {
    3.0/2.0, 0, 0, 0,
    sqrt(3.0)/2.0, sqrt(3.0), 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };
  glLoadMatrixd(matrix);
  //glOrtho(0.0f, UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 0.0f, 0.0f, 1.0f);
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}



int main(int argc, char **argv)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("  Device name: %s\n", prop.name);

  cudaSetDevice(0);

  //initialize automata
  for (int x = 0; x < UNIVERSE_WIDTH; ++x){
    for (int y = 0; y < UNIVERSE_HEIGHT; ++y){
      host_univ[x][y] = thrust::complex<float>(2.0f * (rand() / float(RAND_MAX)) - 1.0f, 2.0f * (rand() / float(RAND_MAX)) - 1.0f);
    }
  }
  
  //host_univ[UNIVERSE_WIDTH/2][UNIVERSE_HEIGHT/2] = make_cuFloatComplex(1.0, 1.0);

  cudaMemcpyToSymbol(univ, host_univ, sizeof(Universe), 0, cudaMemcpyHostToDevice);

  
  



  initOpenGL();

  float scale = 0.4f;

  char title[128];
  cudaEvent_t start, stop;
  float milliseconds = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
      cudaEventRecord(start);
      runAutomata<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1)>>>(true);
      runAutomata<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1)>>>(false);
      cudaEventRecord(stop);

      // rasterize
      rasterizeAutomata1<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1)>>>();

      // rasterize
      rasterizeAutomata2<<<dim3(UNIVERSE_WIDTH, UNIVERSE_HEIGHT, 1), dim3(1,1,1)>>>();

      // copy raster back to host
      cudaMemcpyFromSymbol(host_raster, raster, sizeof(UniImg), 0, cudaMemcpyDeviceToHost);

      //cudaStreamSynchronize(cudaStreamPerThread);

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


      cudaEventElapsedTime((float*)&milliseconds, start, stop);
      sprintf(title, "%.2f executions per sec",  2000.0f / (float) milliseconds);

      glfwSetWindowTitle(window, title);

      glfwPollEvents();

      //Sleep(500);
  }
}