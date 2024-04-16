#pragma once

#include <vector>

#include "IBlurImageGenerator.hpp"

#pragma once

#include <vector>

#include "Homography.hpp"
#include "IBlurImageGenerator.hpp"

class BlurKernelGenerator : public IBlurImageGenerator {
 public:
  enum class Border { ISOLATED, REPLICATE, REFLECT, WRAP };

  BlurKernelGenerator(int aKernelHalfWidth, int aKernelHalfHeight,
                      Border aBorder, float* aBaseImg, int aWidth, int aHeight);

  // bforward: true forward, false backward
  void blurGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                float* BlurImg, float* outputWeight, int width, int height,
                bool bforward) override;
  void blurRgb(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* BlurImgR,
               float* BlurImgG, float* BlurImgB, float* outputWeight, int width,
               int height, bool bforward) override;

  void SetBuffer(int width, int height) override;
  void ClearBuffer() override;

 private:
  int mKernelHalfWidth{};
  int mKernelHalfHeight{};
  Border mBorder{};

  float* mBaseImg{};
  int mWidth{};
  int mHeight{};

  float getKernelWeightedPoint(float* aKernelImg, int xCenter, int yCenter,
                               int aXmin, int aXmax, int aYmin,
                               int aYmax) const;
};