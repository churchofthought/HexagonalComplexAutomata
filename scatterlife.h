
#define UNIVERSE_WIDTH 2*7*7*7
#define UNIVERSE_HEIGHT 2*7*7*7

typedef cuFloatComplex Cell;

typedef cuFloatComplex Universe[UNIVERSE_WIDTH][UNIVERSE_HEIGHT];

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