#pragma once

void warpImageGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                   float* OutputImg, float* outputWeight, int width, int height,
                   float hmatrix[3][3]);

void warpImageRgb(float* InputImgR, float* InputImgG, float* InputImgB,
                  float* inputWeight, int iwidth, int iheight,
                  float* OutputImgR, float* OutputImgG, float* OutputImgB,
                  float* outputWeight, int width, int height,
                  float hmatrix[3][3]);
