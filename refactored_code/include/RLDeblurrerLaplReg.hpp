#pragma once

#include <vector>

#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"
#include "IRegularizer.hpp"

class RLDeblurrerLaplReg {
 public:
  RLDeblurrerLaplReg(IBlurImageGenerator& aBlurGenerator,
                     IErrorCalculator& aErrorCalculator);

  ~RLDeblurrerLaplReg() { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are deblurring algorithm
  ////////////////////////////////////
  void ProjectiveMotionRLDeblurSpsRegGray(float* BlurImg, int iwidth,
                                          int iheight, float* DeblurImg,
                                          int width, int height, int Niter = 20,
                                          bool bPoisson = true,
                                          float lambda = 0.50f);
  void ProjectiveMotionRLDeblurSpsRegRgb(float* BlurImgR, float* BlurImgG,
                                         float* BlurImgB, int iwidth,
                                         int iheight, float* DeblurImgR,
                                         float* DeblurImgG, float* DeblurImgB,
                                         int width, int height, int Niter = 20,
                                         bool bPoisson = true,
                                         float lambda = 0.50f);

  ////////////////////////////////////
  // These functions are used to compute derivatives for regularization
  ////////////////////////////////////
  void ComputeGradientXImageGray(float* Img, int width, int height,
                                 float* DxImg, bool bflag = true);
  void ComputeGradientYImageGray(float* Img, int width, int height,
                                 float* DyImg, bool bflag = true);
  void ComputeGradientImageGray(float* Img, int width, int height, float* DxImg,
                                float* DyImg, bool bflag = true);

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

  // These are buffer and lookup table variables
  float mSpsTable[256]{};

  void SetSpsTable();
  float getSpsWeight(float aValue) const;
};

// Value of lambda used in Levin et al is also un-normalized by minWeight,
// hence it's much smaller The typical range of this Sps in their
// implementation is between 0.001 - 0.004 We use the same lambda as in TV
// regularization for better comparison
class LaplacianRegularizer : public IRegularizer {
 public:
  LaplacianRegularizer();
  ~LaplacianRegularizer() override { ClearBuffer(); }

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
  // These are buffer and lookup table variables
  float mSpsTable[256]{};

  void SetSpsTable();
  float getSpsWeight(float aValue) const;

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