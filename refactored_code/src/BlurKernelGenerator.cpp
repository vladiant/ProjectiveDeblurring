#include "BlurKernelGenerator.hpp"

BlurKernelGenerator::BlurKernelGenerator(int aKernelWidth, int aKernelHeight,
                                         Border aBorder, float* aBaseImg,
                                         int aWidth, int aHeight)
    : mKernelWidth{aKernelWidth},
      mKernelHeight{aKernelHeight},
      mBorder{aBorder},
      mBaseImg{aBaseImg},
      mWidth{aWidth},
      mHeight{aHeight} {}

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

void BlurKernelGenerator::SetBuffer([[maybe_unused]] int width,
                                    [[maybe_unused]] int height) {}

void BlurKernelGenerator::ClearBuffer() {}