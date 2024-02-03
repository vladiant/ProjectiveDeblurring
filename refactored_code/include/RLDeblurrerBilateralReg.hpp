#pragma once

#include <vector>

#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"

class RLDeblurrerBilateralReg {
 public:
  RLDeblurrerBilateralReg(IBlurImageGenerator& aBlurGenerator,
                          IErrorCalculator& aErrorCalculator);

  ~RLDeblurrerBilateralReg() { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are deblurring algorithm
  ////////////////////////////////////

  // We use the same lambda as in TV regularization for better comparison
  // Parameter setting, noise variance, for bilateral reg is the same as
  // laplacian reg.
  void ProjectiveMotionRLDeblurBilateralRegGray(
      float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
      int height, int Niter = 20, bool bPoisson = true, float lambda = 0.50f);
  void ProjectiveMotionRLDeblurBilateralRegRgb(
      float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth,
      int iheight, float* DeblurImgR, float* DeblurImgG, float* DeblurImgB,
      int width, int height, int Niter = 20, bool bPoisson = true,
      float lambda = 0.50f);

  ////////////////////////////////////
  // These functions are used to compute derivatives for regularization
  ////////////////////////////////////
  void ComputeBilaterRegImageGray(float* Img, int width, int height,
                                  float* BRImg);

  void SetBilateralTable();

 private:
  IBlurImageGenerator& mBlurGenerator;
  IErrorCalculator& mErrorCalculator;

  // These are buffer and lookup table variables
  float mBilateralTable[256]{};

  std::vector<float> mBlurImgBuffer;
  std::vector<float> mBlurImgBufferR;
  std::vector<float> mBlurImgBufferG;
  std::vector<float> mBlurImgBufferB;

  std::vector<float> mBlurWeightBuffer;

  std::vector<float> mErrorImgBuffer;
  std::vector<float> mErrorImgBufferR;
  std::vector<float> mErrorImgBufferG;
  std::vector<float> mErrorImgBufferB;

  std::vector<float> mErrorWeightBuffer;
};