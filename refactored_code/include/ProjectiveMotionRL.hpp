#pragma once

#include <cmath>
#include <cstring>

#include "Homography.hpp"
#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"

class ProjectiveMotionRL {
 public:
  ProjectiveMotionRL(IBlurImageGenerator& aBlurGenerator,
                     IErrorCalculator& aErrorCalculator);

  ~ProjectiveMotionRL() { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  void SetSpsTable() {
    int i, t = 1;
    // Parameters, e.g. noiseVar and epilson, are set according to Levin et al
    // Siggraph'07
    float powD = 0.8f, noiseVar = 0.005f, epilson = t / 255.0f;
    float minWeight =
        exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

    for (i = 0; i <= t; i++) {
      mSpsTable[i] = 1.0f;
    }
    for (i = t + 1; i < 256; i++) {
      mSpsTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                      pow(i / 255.0f, powD - 1.0f)) /
                     minWeight;

      // Reweighting scheme from Levin et al Siggraph'07,
      // Similar Effect, smaller smoothing weight for large gradient
      // float minEpi = pow(epilson, powD - 2.0f);
      // mSpsTable[i] = pow(i / 255.0f, powD - 2.0f) / minEpi;
    }
  }

  void SetBilateralTable() {
    int i;
    // Parameters are set according to Levin et al Siggraph'07
    // Better result can be obtained by using smaller noiseVar, but for
    // fairness, we use the same setting.
    float noiseVar = 0.005f;

    // Standard Bilateral Weight
    for (i = 0; i < 256; i++) {
      mBilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
    }

    // Bilateral Laplician Regularization
    // int t = 1;
    // float powD = 0.8f;
    // float epilson = t / 255.0f;
    // float minWeight =
    //     exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);
    // for (i = 0; i <= t; i++) {
    //   mBilateralTable[i] = 1.0f;
    // }
    // for (i = t + 1; i < 256; i++) {
    //   mBilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
    //                        pow(i / 255.0f, powD - 1.0f)) /
    //                       minWeight;
    // }
  }

  ////////////////////////////////////
  // These functions are deblurring algorithm
  ////////////////////////////////////

  // Value of lambda used in Levin et al is also un-normalized by minWeight,
  // hence it's much smaller The typical range of this Sps in their
  // implementation is between 0.001 - 0.004 We use the same lambda as in TV
  // regularization for better comparison
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
  // This is the bilateral laplacian regularization
  void ProjectiveMotionRLDeblurBilateralLapRegGray(
      float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
      int height, int Niter = 20, bool bPoisson = true, float lambda = 0.50f);
  void ProjectiveMotionRLDeblurBilateralLapRegRgb(
      float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth,
      int iheight, float* DeblurImgR, float* DeblurImgG, float* DeblurImgB,
      int width, int height, int Niter = 20, bool bPoisson = true,
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
  void ComputeBilaterRegImageGray(float* Img, int width, int height,
                                  float* BRImg);

 private:
  float getSpsWeight(float aValue) const;

  IBlurImageGenerator& mBlurGenerator;
  IErrorCalculator& mErrorCalculator;

  // These are buffer and lookup table variables
  float mBilateralTable[256];
  float mSpsTable[256];
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
