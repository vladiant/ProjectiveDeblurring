#pragma once

class IBlurImageGenerator {
 public:
  virtual ~IBlurImageGenerator() = default;

  // bforward: true forward, false backward
  virtual void blurGray(float* InputImg, float* inputWeight, int iwidth,
                        int iheight, float* BlurImg, float* outputWeight,
                        int width, int height, bool bforward) = 0;
  virtual void blurRgb(float* InputImgR, float* InputImgG, float* InputImgB,
                       float* inputWeight, int iwidth, int iheight,
                       float* BlurImgR, float* BlurImgG, float* BlurImgB,
                       float* outputWeight, int width, int height,
                       bool bforward) = 0;

  virtual void SetBuffer(int width, int height) = 0;
  virtual void ClearBuffer() = 0;
};
