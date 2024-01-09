#include "warping.h"

#include "BicubicInterpolation.h"

void warpImage(float* InputImg, float* inputWeight, int iwidth, int iheight,
               float* OutputImg, float* outputWeight, int width, int height,
               const Homography& homography) {
  const float woffset = width * 0.5f;
  const float hoffset = height * 0.5f;
  const float iwoffset = iwidth * 0.5f;
  const float ihoffset = iheight * 0.5f;

  for (int y = 0, index = 0; y < height; y++) {
    for (int x = 0; x < width; x++, index++) {
      float fx = x - woffset;
      float fy = y - hoffset;
      homography.Transform(fx, fy);  // Inverse mapping, use inverse instead
      fx += iwoffset;
      fy += ihoffset;

      if (fx >= 0 && fx < iwidth - 1 && fy >= 0 && fy < iheight - 1) {
        if (inputWeight) {
          outputWeight[index] =
              0.01f +
              ReturnInterpolatedValueFast(fx, fy, inputWeight, iwidth, iheight);
        } else {
          outputWeight[index] = 1.01f;
        }
      } else {
        outputWeight[index] = 0.01f;
      }

      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx >= iwidth - 1.001f) fx = iwidth - 1.001f;
      if (fy >= iheight - 1.001f) fy = iheight - 1.001f;

      OutputImg[index] =
          ReturnInterpolatedValueFast(fx, fy, InputImg, iwidth, iheight);
    }
  }
}

void warpImage(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* OutputImgR,
               float* OutputImgG, float* OutputImgB, float* outputWeight,
               int width, int height, const Homography& homography) {
  const float woffset = width * 0.5f;
  const float hoffset = height * 0.5f;
  const float iwoffset = iwidth * 0.5f;
  const float ihoffset = iheight * 0.5f;

  for (int y = 0, index = 0; y < height; y++) {
    for (int x = 0; x < width; x++, index++) {
      float fx = x - woffset;
      float fy = y - hoffset;
      homography.Transform(fx, fy);  // Inverse mapping, use inverse instead
      fx += iwoffset;
      fy += ihoffset;

      if (fx >= 0 && fx < iwidth - 1 && fy >= 0 && fy < iheight - 1) {
        if (inputWeight) {
          outputWeight[index] =
              0.01f +
              ReturnInterpolatedValueFast(fx, fy, inputWeight, iwidth, iheight);
        } else {
          outputWeight[index] = 1.01f;
        }
      } else {
        outputWeight[index] = 0.01f;
      }

      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx >= iwidth - 1.001f) fx = iwidth - 1.001f;
      if (fy >= iheight - 1.001f) fy = iheight - 1.001f;

      ReturnInterpolatedValueFast(fx, fy, InputImgR, InputImgG, InputImgB,
                                  iwidth, iheight, OutputImgR[index],
                                  OutputImgG[index], OutputImgB[index]);
    }
  }
}