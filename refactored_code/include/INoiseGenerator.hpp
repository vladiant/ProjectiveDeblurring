#pragma once

class INoiseGenerator {
 public:
  virtual ~INoiseGenerator() = default;

  ////////////////////////////////////
  // These functions are used to generate noise
  ////////////////////////////////////
  virtual void addNoiseGray(float* Img, int width, int height,
                            float* aOutImg) = 0;

  virtual void addNoiseRgb(float* ImgR, float* ImgG, float* ImgB, int width,
                           int height, float* aOutImgR, float* aOutImgG,
                           float* aOutImgB) = 0;
};