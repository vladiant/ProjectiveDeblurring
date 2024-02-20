#pragma once

#include <vector>

#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"
#include "IRegularizer.hpp"

// The lambda in TV regularization is 0.002, but it's un-normalized weight
// Intensity range is between 0 and 1, so, the actual weight is 0.002f * 255 =
// 0.51f for normalized weight
class TVRegularizer : public IRegularizer {
 public:
  ~TVRegularizer() override { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are used to apply regularization
  ////////////////////////////////////
  void applyRegularizationGray(float* DeblurImg, int width, int height,
                               bool bPoisson, float lambda) override;

  void applyRegularizationRgb(float* DeblurImgR, float* DeblurImgG,
                              float* DeblurImgB, int width, int height,
                              bool bPoisson, float lambda) override;

 private:
  ////////////////////////////////////
  // These functions are used to compute derivatives for regularization
  ////////////////////////////////////
  void ComputeGradientXImageGray(float* Img, int width, int height,
                                 float* DxImg, bool bflag = true);
  void ComputeGradientYImageGray(float* Img, int width, int height,
                                 float* DyImg, bool bflag = true);
  void ComputeGradientImageGray(float* Img, int width, int height, float* DxImg,
                                float* DyImg, bool bflag = true);

  std::vector<float> mDxImg;
  std::vector<float> mDyImg;
  std::vector<float> mDxxImg;
  std::vector<float> mDyyImg;

  std::vector<float> mDxImgR;
  std::vector<float> mDyImgR;
  std::vector<float> mDxxImgR;
  std::vector<float> mDyyImgR;
  std::vector<float> mDxImgG;
  std::vector<float> mDyImgG;
  std::vector<float> mDxxImgG;
  std::vector<float> mDyyImgG;
  std::vector<float> mDxImgB;
  std::vector<float> mDyImgB;
  std::vector<float> mDxxImgB;
  std::vector<float> mDyyImgB;
};
