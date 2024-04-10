#include "BlurKernelGenerator.hpp"

void BlurKernelGenerator::blurGray(float* InputImg, float* inputWeight,
                                   int iwidth, int iheight, float* BlurImg,
                                   float* outputWeight, int width, int height,
                                   bool bforward) {}

void BlurKernelGenerator::blurRgb(float* InputImgR, float* InputImgG,
                                  float* InputImgB, float* inputWeight,
                                  int iwidth, int iheight, float* BlurImgR,
                                  float* BlurImgG, float* BlurImgB,
                                  float* outputWeight, int width, int height,
                                  bool bforward) {}

void BlurKernelGenerator::SetBuffer(int width, int height) {}

void BlurKernelGenerator::ClearBuffer() {}