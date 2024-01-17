#pragma once

void bicubicweight(float x, float y, float (&w)[4]);
void bicubicweightFast(float x, float y, float (&w)[4]);
float ReturnInterpolatedValue(float x, float y, float* img, int width,
                              int height);
float ReturnInterpolatedValueFast(float x, float y, float* img, int width,
                                  int height);
void ReturnInterpolatedValueFast(float x, float y, float* RGBimg, int width,
                                 int height, float& R, float& G, float& B);
