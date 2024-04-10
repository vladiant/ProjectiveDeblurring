#pragma once

#include <vector>

#include "IBlurImageGenerator.hpp"

#pragma once

#include <vector>

#include "Homography.hpp"
#include "IBlurImageGenerator.hpp"

class BlurKernelGenerator : public IBlurImageGenerator {
 public:
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
};