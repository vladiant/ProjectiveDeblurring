#pragma once

#include "homography.h"

void warpImageGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                   float* OutputImg, float* outputWeight, int width, int height,
                   const Homography& homography);

void warpImageRgb(float* InputImgRGB, float* inputWeight, int iwidth,
                  int iheight, float* OutputImgRGB, float* outputWeight,
                  int width, int height, const Homography& homography);