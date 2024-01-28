#pragma once

#include "IErrorCalculator.hpp"

class EmptyErrorCalculator : public IErrorCalculator {
 public:
  ~EmptyErrorCalculator() override = default;
  float calculateErrorRgb(float* ImgR, float* ImgG, float* ImgB, int width,
                          int height) override;

  float calculateErrorGray(float* Img, int width, int height) override;
};