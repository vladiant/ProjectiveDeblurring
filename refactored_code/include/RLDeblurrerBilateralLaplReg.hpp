#pragma once

#include <cmath>
#include <cstring>

#include "Homography.hpp"
#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"

class RLDeblurrerBilateralLaplReg {
 public:
  RLDeblurrerBilateralLaplReg(IBlurImageGenerator& aBlurGenerator,
                              IErrorCalculator& aErrorCalculator);

  ~RLDeblurrerBilateralLaplReg() { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are deblurring algorithm
  ////////////////////////////////////

  // This is the bilateral laplacian regularization
  void ProjectiveMotionRLDeblurBilateralLapRegGray(
      float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
      int height, int Niter = 20, bool bPoisson = true, float lambda = 0.50f);
  void ProjectiveMotionRLDeblurBilateralLapRegRgb(
      float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth,
      int iheight, float* DeblurImgR, float* DeblurImgG, float* DeblurImgB,
      int width, int height, int Niter = 20, bool bPoisson = true,
      float lambda = 0.50f);

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

  void ProcessGray(float* BlurImg, int iwidth, int iheight, float* DeblurImg,
                   int width, int height, int Niter = 20, bool bPoisson = true,
                   float lambda = 0.50f);
  void ProcessRgb(float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth,
                  int iheight, float* DeblurImgR, float* DeblurImgG,
                  float* DeblurImgB, int width, int height, int Niter = 20,
                  bool bPoisson = true, float lambda = 0.50f);

  void SetBilateralTable();

  ////////////////////////////////////
  // These functions are used to compute derivatives for regularization
  ////////////////////////////////////
  void ComputeBilaterRegImageGray(float* Img, int width, int height,
                                  float* BRImg);
};
