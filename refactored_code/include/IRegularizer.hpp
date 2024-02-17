#pragma once

class IRegularizer {
 public:
  virtual ~IRegularizer() = default;

  ////////////////////////////////////
  // These functions are used to apply regularization
  ////////////////////////////////////
  virtual void applyRegularizationGray(float* DeblurImg, int width, int height,
                                       bool bPoisson, float lambda) = 0;

  virtual void applyRegularizationRgb(float* DeblurImgR, float* DeblurImgG,
                                      float* DeblurImgB, int width, int height,
                                      bool bPoisson, float lambda) = 0;
};