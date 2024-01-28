#pragma once

#include "Homography.hpp"

void warpImageGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                   float* OutputImg, float* outputWeight, int width, int height,
                   const Homography& homography);

void warpImageRgb(float* InputImgR, float* InputImgG, float* InputImgB,
                  float* inputWeight, int iwidth, int iheight,
                  float* OutputImgR, float* OutputImgG, float* OutputImgB,
                  float* outputWeight, int width, int height,
                  const Homography& homography);