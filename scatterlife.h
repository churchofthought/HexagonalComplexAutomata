
#define UNIVERSE_WIDTH 128
#define UNIVERSE_HEIGHT 128

typedef float UniverseDevice[UNIVERSE_WIDTH][UNIVERSE_HEIGHT][2];
typedef thrust::complex<float> Universe[UNIVERSE_WIDTH][UNIVERSE_HEIGHT];

struct RGB
{
  float R;
  float G;
  float B;
};

struct HSL
{
  float H;
  float S;
  float L;
};

#define RASTER_UPSAMPLE 1
#define RASTER_WIDTH (RASTER_UPSAMPLE * UNIVERSE_WIDTH)
#define RASTER_HEIGHT (RASTER_UPSAMPLE * UNIVERSE_HEIGHT)
typedef RGB UniImg[RASTER_WIDTH][RASTER_HEIGHT];

//#define INITIAL_PARTICLE_COUNT (UNIVERSE_WIDTH*UNIVERSE_HEIGHT*0.05)