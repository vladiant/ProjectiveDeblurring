#include "BicubicInterpolation.h"

#include <cmath>

void bicubicweight(float x, float y, float (&w)[4]) {
  float ww = NAN;

  const float ix = x - (int)(x), iy = y - (int)(y);
  if (x >= 0 && y >= 0) {
    w[0] = (1 - ix) * (1 - iy);
    w[1] = (1 - ix) * (iy);
    w[2] = (ix) * (1 - iy);
    w[3] = (ix) * (iy);
    ww = w[0] + w[1] + w[2] + w[3];
  } else if (x >= 0) {
    w[0] = (1 - ix) * (iy + 1);
    w[1] = (1 - ix) * (-iy);
    w[2] = (ix) * (iy + 1);
    w[3] = (ix) * (-iy);
    ww = w[0] + w[1] + w[2] + w[3];
  } else if (y >= 0) {
    w[0] = (ix + 1) * (1 - iy);
    w[1] = (ix + 1) * (iy);
    w[2] = (-ix) * (1 - iy);
    w[3] = (-ix) * (iy);
    ww = w[0] + w[1] + w[2] + w[3];
  } else {
    w[0] = (ix + 1) * (iy + 1);
    w[1] = (ix + 1) * (-iy);
    w[2] = (-ix) * (iy + 1);
    w[3] = (-ix) * (-iy);
    ww = w[0] + w[1] + w[2] + w[3];
  }

  w[0] /= ww;
  w[1] /= ww;
  w[2] /= ww;
  w[3] /= ww;
}

void bicubicweightFast(float x, float y, float (&w)[4]) {
  float ix = x - (int)(x), iy = y - (int)(y);
  w[1] = (1.0f - ix) * iy;
  w[2] = ix * (1.0f - iy);
  w[3] = ix * iy;
  w[0] = 1.0f - w[1] - w[2] - w[3];
}

float ReturnInterpolatedValue(float x, float y, float* img, int width,
                              int /*height*/) {
  float w[4], value = NAN;
  int ix = (int)(x), iy = (int)(y);
  int index = iy * width + ix;
  bicubicweight(x, y, w);

  value = img[index] * w[0] + img[index + width] * w[1] +
          img[index + 1] * w[2] + img[index + width + 1] * w[3];

  return value;
}

float ReturnInterpolatedValueFast(float x, float y, float* img, int width,
                                  int /*height*/) {
  float w[4];
  int ix = (int)(x), iy = (int)(y);
  int index = iy * width + ix;
  float fx = x - ix, fy = y - iy;
  w[1] = (1.0f - fx) * fy;
  w[2] = fx * (1.0f - fy);
  w[3] = fx * fy;
  w[0] = 1.0f - w[1] - w[2] - w[3];

  return img[index] * w[0] + img[index + width] * w[1] + img[index + 1] * w[2] +
         img[index + width + 1] * w[3];
}

void ReturnInterpolatedValueFast(float x, float y, float* RGBimg, int width,
                                 int /*height*/, float& R, float& G, float& B) {
  float w[4];
  int ix = (int)(x), iy = (int)(y);
  int index = iy * width + ix;
  float fx = x - ix, fy = y - iy;
  w[1] = (1.0f - fx) * fy;
  w[2] = fx * (1.0f - fy);
  w[3] = fx * fy;
  w[0] = 1.0f - w[1] - w[2] - w[3];

  R = RGBimg[3 * index] * w[0] + RGBimg[3 * (index + width)] * w[1] +
      RGBimg[3 * (index + 1)] * w[2] + RGBimg[3 * (index + width + 1)] * w[3];
  G = RGBimg[3 * index + 1] * w[0] + RGBimg[3 * (index + width) + 1] * w[1] +
      RGBimg[3 * (index + 1) + 1] * w[2] +
      RGBimg[3 * (index + width + 1) + 1] * w[3];
  B = RGBimg[3 * index + 2] * w[0] + RGBimg[3 * (index + width) + 2] * w[1] +
      RGBimg[3 * (index + 1) + 2] * w[2] +
      RGBimg[3 * (index + width + 1) + 2] * w[3];
}