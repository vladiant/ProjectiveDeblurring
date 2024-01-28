#pragma once

#include <vector>

#include "IBlurImageGenerator.h"
#include "IErrorCalculator.h"

class RLDeblurrer {
public:
  RLDeblurrer(IBlurImageGenerator& aBlurGenerator,
                     IErrorCalculator& aErrorCalculator);

  ~RLDeblurrer() { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are deblurring algorithm
  ////////////////////////////////////
  // This is the Basic algorithm
  // DeblurImg: the Input itself is initialization, so you can load
  // yBilateralLap own initialization
  void ProjectiveMotionRLDeblurGray(float* BlurImg, int iwidth, int iheight,
                                    float* DeblurImg, int width, int height,
                                    int Niter = 20, bool bPoisson = true);
  void ProjectiveMotionRLDeblurRgb(float* BlurImgR, float* BlurImgG,
                                   float* BlurImgB, int iwidth, int iheight,
                                   float* DeblurImgR, float* DeblurImgG,
                                   float* DeblurImgB, int width, int height,
                                   int Niter = 20, bool bPoisson = true);

private:
  IBlurImageGenerator& mBlurGenerator;
  IErrorCalculator& mErrorCalculator;

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