#pragma once

#include <cmath>
#include <cstring>
#include <random>

#include "IBlurImageGenerator.h"
#include "homography.h"

//#define __SHOWERROR__

class ProjectiveMotionRL {
 public:
  // Random engine seed
  constexpr static int kSeed = 1234;

  ProjectiveMotionRL(IBlurImageGenerator& aBlurGenerator);

  ~ProjectiveMotionRL() {
    ClearBuffer();
    mGroundTruthImg.clear();
    mGroundTruthImgR.clear();
    mGroundTruthImgG.clear();
    mGroundTruthImgB.clear();
  }

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
  void SetGroundTruthImgGray(float* GroundTruth, int width, int height) {
    mGroundTruthImg.resize(width * height);
    memcpy(mGroundTruthImg.data(), GroundTruth, width * height * sizeof(float));
  }
  void SetGroundTruthImgRgb(float* GroundTruthR, float* GroundTruthG,
                            float* GroundTruthB, int width, int height) {
    mGroundTruthImgR.resize(width * height);
    mGroundTruthImgG.resize(width * height);
    mGroundTruthImgB.resize(width * height);
    memcpy(mGroundTruthImgR.data(), GroundTruthR,
           width * height * sizeof(float));
    memcpy(mGroundTruthImgG.data(), GroundTruthG,
           width * height * sizeof(float));
    memcpy(mGroundTruthImgB.data(), GroundTruthB,
           width * height * sizeof(float));
  }

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

  // This are the deblurring algorithm with regularization
  // Details please refers to paper
  // The lambda in TV regularization is 0.002, but it's un-normalized weight
  // Intensity range is between 0 and 1, so, the actual weight is 0.002f * 255 =
  // 0.51f for normalized weight
  void ProjectiveMotionRLDeblurTVRegGray(float* BlurImg, int iwidth,
                                         int iheight, float* DeblurImg,
                                         int width, int height, int Niter = 20,
                                         bool bPoisson = true,
                                         float lambda = 0.50f);
  void ProjectiveMotionRLDeblurTVRegRgb(float* BlurImgR, float* BlurImgG,
                                        float* BlurImgB, int iwidth,
                                        int iheight, float* DeblurImgR,
                                        float* DeblurImgG, float* DeblurImgB,
                                        int width, int height, int Niter = 20,
                                        bool bPoisson = true,
                                        float lambda = 0.50f);
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

  ////////////////////////////////////
  // These functions are used to compute Errors
  ////////////////////////////////////
  float ComputeRMSErrorGray(float* GroundTruth, float* DeblurredImg, int width,
                            int height) {
    float RMS = 0;
    if (GroundTruth) {
      int x, y, index;
      for (y = 0, index = 0; y < height; y++) {
        for (x = 0; x < width; x++, index++) {
          RMS += (GroundTruth[index] - DeblurredImg[index]) *
                 (GroundTruth[index] - DeblurredImg[index]);
        }
      }
    }
    // The RMS error output in paper have been multiplied by 255
    return sqrt(RMS / (width * height));
  }

  ////////////////////////////////////
  // These functions are used to generate noise
  ////////////////////////////////////
  // Normal random number generator, variance = 1
  float normalrand() {
    std::normal_distribution<float> normalDist(0.0f, 1.0f);
    // float val = 0;
    // for (int i = 0; i != 12; ++i) val += ((float)(rand()) / RAND_MAX);
    // return val - 6.0f;
    return normalDist(mRandomEngine);
  }

  // Noise variance = amp
  void gaussianNoiseGray(float* Img, int width, int height, float amp) {
    int x, y, index;
    float random, noise;
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        random = normalrand() / 255.0f;
        noise = amp * random;
        Img[index] += noise;
        if (Img[index] > 1.0f) Img[index] = 1.0f;
        if (Img[index] < 0.0f) Img[index] = 0.0f;
      }
    }
  }

 private:
  float getSpsWeight(float aValue) const;

  IBlurImageGenerator& mBlurGenerator;

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

  // This variables are used for RMS computation
  std::vector<float> mGroundTruthImg;
  std::vector<float> mGroundTruthImgR;
  std::vector<float> mGroundTruthImgG;
  std::vector<float> mGroundTruthImgB;

  // Random values generation
  std::random_device mRandomDevice;
  std::mt19937 mRandomEngine{mRandomDevice()};
};
