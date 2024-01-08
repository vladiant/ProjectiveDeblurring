#pragma once

#include "homography.h"

void warpImage(float* InputImg, float* inputWeight, int iwidth, int iheight,
               float* OutputImg, float* outputWeight, int width, int height,
               const Homography& homography);

void warpImage(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* OutputImgR,
               float* OutputImgG, float* OutputImgB, float* outputWeight,
               int width, int height, const Homography& homography);