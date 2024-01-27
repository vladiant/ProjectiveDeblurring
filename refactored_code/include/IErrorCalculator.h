#pragma once

class IErrorCalculator {
 public:
  virtual ~IErrorCalculator() = default;
  ////////////////////////////////////
  // These functions are used to compute Errors
  ////////////////////////////////////
  virtual float calculateErrorGray(float* Img, int width, int height) = 0;

  virtual float calculateErrorRgb(float* ImgR, float* ImgG, float* ImgB,
                                  int width, int height) = 0;
};