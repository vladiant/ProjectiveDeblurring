#pragma once

#include <cmath>
#include <cstring>
#include <random>

#include "homography.h"

//#define __SHOWERROR__

class ProjectiveMotionRL {
 public:
  constexpr static int NumSamples = 30;
  // Random engine seed
  constexpr static int kSeed = 1234;

  ProjectiveMotionRL() {
    mRandomEngine.seed(kSeed);
    int i;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = 1;
      Hmatrix[i].Hmatrix[0][1] = 0;
      Hmatrix[i].Hmatrix[0][2] = 0;
      Hmatrix[i].Hmatrix[1][0] = 0;
      Hmatrix[i].Hmatrix[1][1] = 1;
      Hmatrix[i].Hmatrix[1][2] = 0;
      Hmatrix[i].Hmatrix[2][0] = 0;
      Hmatrix[i].Hmatrix[2][1] = 0;
      Hmatrix[i].Hmatrix[2][2] = 1;
      IHmatrix[i].Hmatrix[0][0] = 1;
      IHmatrix[i].Hmatrix[0][1] = 0;
      IHmatrix[i].Hmatrix[0][2] = 0;
      IHmatrix[i].Hmatrix[1][0] = 0;
      IHmatrix[i].Hmatrix[1][1] = 1;
      IHmatrix[i].Hmatrix[1][2] = 0;
      IHmatrix[i].Hmatrix[2][0] = 0;
      IHmatrix[i].Hmatrix[2][1] = 0;
      IHmatrix[i].Hmatrix[2][2] = 1;
    }

    SetSpsTable();
    SetBilateralTable();
  }

  ~ProjectiveMotionRL() {
    ClearBuffer();
    GroundTruthImg.clear();
    GroundTruthImgR.clear();
    GroundTruthImgG.clear();
    GroundTruthImgB.clear();
  }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height) {
    mInitialInputWeight.assign(width * height, 1.01);

    WarpImgBuffer.resize(width * height);
    WarpImgBufferR.resize(width * height);
    WarpImgBufferG.resize(width * height);
    WarpImgBufferB.resize(width * height);
    WarpWeightBuffer.resize(width * height);
    BlurImgBuffer.resize(width * height);
    BlurImgBufferR.resize(width * height);
    BlurImgBufferG.resize(width * height);
    BlurImgBufferB.resize(width * height);
    BlurWeightBuffer.resize(width * height);
    ErrorImgBuffer.resize(width * height);
    ErrorImgBufferR.resize(width * height);
    ErrorImgBufferG.resize(width * height);
    ErrorImgBufferB.resize(width * height);
    ErrorWeightBuffer.resize(width * height);
  }
  void ClearBuffer() {
    WarpImgBuffer.clear();

    WarpImgBufferR.clear();

    WarpImgBufferG.clear();

    WarpImgBufferB.clear();

    WarpWeightBuffer.clear();

    BlurImgBuffer.clear();

    BlurImgBufferR.clear();

    BlurImgBufferG.clear();

    BlurImgBufferB.clear();

    BlurWeightBuffer.clear();

    ErrorImgBuffer.clear();

    ErrorImgBufferR.clear();

    ErrorImgBufferG.clear();

    ErrorImgBufferB.clear();

    ErrorWeightBuffer.clear();
  }

  void SetSpsTable() {
    int i, t = 1;
    // Parameters, e.g. noiseVar and epilson, are set according to Levin et al
    // Siggraph'07
    float powD = 0.8f, noiseVar = 0.005f, epilson = t / 255.0f;
    float minWeight =
        exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

    for (i = 0; i <= t; i++) {
      SpsTable[i] = 1.0f;
    }
    for (i = t + 1; i < 256; i++) {
      SpsTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                     pow(i / 255.0f, powD - 1.0f)) /
                    minWeight;

      // Reweighting scheme from Levin et al Siggraph'07,
      // Similar Effect, smaller smoothing weight for large gradient
      // float minEpi = pow(epilson, powD - 2.0f);
      // SpsTable[i] = pow(i / 255.0f, powD - 2.0f) / minEpi;
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
      BilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
    }

    // Bilateral Laplician Regularization
    // int t = 1;
    // float powD = 0.8f;
    // float epilson = t / 255.0f;
    // float minWeight =
    //     exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);
    // for (i = 0; i <= t; i++) {
    //   BilateralTable[i] = 1.0f;
    // }
    // for (i = t + 1; i < 256; i++) {
    //   BilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
    //                        pow(i / 255.0f, powD - 1.0f)) /
    //                       minWeight;
    // }
  }
  void SetGroundTruthImgGray(float* GroundTruth, int width, int height) {
    GroundTruthImg.resize(width * height);
    memcpy(GroundTruthImg.data(), GroundTruth, width * height * sizeof(float));
  }
  void SetGroundTruthImgRgb(float* GroundTruthR, float* GroundTruthG,
                            float* GroundTruthB, int width, int height) {
    GroundTruthImgR.resize(width * height);
    GroundTruthImgG.resize(width * height);
    GroundTruthImgB.resize(width * height);
    memcpy(GroundTruthImgR.data(), GroundTruthR,
           width * height * sizeof(float));
    memcpy(GroundTruthImgG.data(), GroundTruthG,
           width * height * sizeof(float));
    memcpy(GroundTruthImgB.data(), GroundTruthB,
           width * height * sizeof(float));
  }

  ////////////////////////////////////
  // These functions are used to set the homography
  ////////////////////////////////////
  void SetHomography(Homography H, int i) {
    if (i >= 0 && i < NumSamples) {
      memcpy(Hmatrix[i].Hmatrix[0], H.Hmatrix[0], 3 * sizeof(float));
      memcpy(Hmatrix[i].Hmatrix[1], H.Hmatrix[1], 3 * sizeof(float));
      memcpy(Hmatrix[i].Hmatrix[2], H.Hmatrix[2], 3 * sizeof(float));

      Homography::MatrixInverse(Hmatrix[i].Hmatrix, IHmatrix[i].Hmatrix);
    }
  }

  void SetGlobalRotation(float degree) {
    int i;
    float deltadegree = (degree * M_PI / 180.0f) / NumSamples;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = cos(deltadegree * i);
      Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
      Hmatrix[i].Hmatrix[0][2] = 0;
      Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
      Hmatrix[i].Hmatrix[1][1] = cos(deltadegree * i);
      Hmatrix[i].Hmatrix[1][2] = 0;
      Hmatrix[i].Hmatrix[2][0] = 0;
      Hmatrix[i].Hmatrix[2][1] = 0;
      Hmatrix[i].Hmatrix[2][2] = 1;
      IHmatrix[i].Hmatrix[0][0] = cos(deltadegree * i);
      IHmatrix[i].Hmatrix[0][1] = -sin(deltadegree * i);
      IHmatrix[i].Hmatrix[0][2] = 0;
      IHmatrix[i].Hmatrix[1][0] = sin(deltadegree * i);
      IHmatrix[i].Hmatrix[1][1] = cos(deltadegree * i);
      IHmatrix[i].Hmatrix[1][2] = 0;
      IHmatrix[i].Hmatrix[2][0] = 0;
      IHmatrix[i].Hmatrix[2][1] = 0;
      IHmatrix[i].Hmatrix[2][2] = 1;
    }
  }
  void SetGlobalScaling(float scalefactor) {
    int i;
    float deltascale = (scalefactor - 1.0f) / NumSamples;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = 1.0f + i * deltascale;
      Hmatrix[i].Hmatrix[0][1] = 0;
      Hmatrix[i].Hmatrix[0][2] = 0;
      Hmatrix[i].Hmatrix[1][0] = 0;
      Hmatrix[i].Hmatrix[1][1] = 1.0f + i * deltascale;
      Hmatrix[i].Hmatrix[1][2] = 0;
      Hmatrix[i].Hmatrix[2][0] = 0;
      Hmatrix[i].Hmatrix[2][1] = 0;
      Hmatrix[i].Hmatrix[2][2] = 1;
      IHmatrix[i].Hmatrix[0][0] = 1.0f / (1.0f + i * deltascale);
      IHmatrix[i].Hmatrix[0][1] = 0;
      IHmatrix[i].Hmatrix[0][2] = 0;
      IHmatrix[i].Hmatrix[1][0] = 0;
      IHmatrix[i].Hmatrix[1][1] = 1.0f / (1.0f + i * deltascale);
      IHmatrix[i].Hmatrix[1][2] = 0;
      IHmatrix[i].Hmatrix[2][0] = 0;
      IHmatrix[i].Hmatrix[2][1] = 0;
      IHmatrix[i].Hmatrix[2][2] = 1;
    }
  }
  void SetGlobalTranslation(float dx, float dy) {
    int i;
    float deltadx = dx / NumSamples;
    float deltady = dy / NumSamples;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = 1;
      Hmatrix[i].Hmatrix[0][1] = 0;
      Hmatrix[i].Hmatrix[0][2] = i * deltadx;
      Hmatrix[i].Hmatrix[1][0] = 0;
      Hmatrix[i].Hmatrix[1][1] = 1;
      Hmatrix[i].Hmatrix[1][2] = i * deltady;
      Hmatrix[i].Hmatrix[2][0] = 0;
      Hmatrix[i].Hmatrix[2][1] = 0;
      Hmatrix[i].Hmatrix[2][2] = 1;
      IHmatrix[i].Hmatrix[0][0] = 1;
      IHmatrix[i].Hmatrix[0][1] = 0;
      IHmatrix[i].Hmatrix[0][2] = -i * deltadx;
      IHmatrix[i].Hmatrix[1][0] = 0;
      IHmatrix[i].Hmatrix[1][1] = 1;
      IHmatrix[i].Hmatrix[1][2] = -i * deltady;
      IHmatrix[i].Hmatrix[2][0] = 0;
      IHmatrix[i].Hmatrix[2][1] = 0;
      IHmatrix[i].Hmatrix[2][2] = 1;
    }
  }
  void SetGlobalPerspective(float px, float py) {
    int i;
    float deltapx = px / NumSamples;
    float deltapy = py / NumSamples;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = 1;
      Hmatrix[i].Hmatrix[0][1] = 0;
      Hmatrix[i].Hmatrix[0][2] = 0;
      Hmatrix[i].Hmatrix[1][0] = 0;
      Hmatrix[i].Hmatrix[1][1] = 1;
      Hmatrix[i].Hmatrix[1][2] = 0;
      Hmatrix[i].Hmatrix[2][0] = i * deltapx;
      Hmatrix[i].Hmatrix[2][1] = i * deltapy;
      Hmatrix[i].Hmatrix[2][2] = 1;
      IHmatrix[i].Hmatrix[0][0] = 1;
      IHmatrix[i].Hmatrix[0][1] = 0;
      IHmatrix[i].Hmatrix[0][2] = 0;
      IHmatrix[i].Hmatrix[1][0] = 0;
      IHmatrix[i].Hmatrix[1][1] = 1;
      IHmatrix[i].Hmatrix[1][2] = 0;
      IHmatrix[i].Hmatrix[2][0] = -i * deltapx;
      IHmatrix[i].Hmatrix[2][1] = -i * deltapy;
      IHmatrix[i].Hmatrix[2][2] = 1;
    }
  }

  void SetGlobalParameters(float degree, float scalefactor, float px, float py,
                           float dx, float dy) {
    int i;
    float deltadegree = (degree * M_PI / 180.0f) / NumSamples;
    float deltascale = (scalefactor - 1.0f) / NumSamples;
    float deltapx = px / NumSamples;
    float deltapy = py / NumSamples;
    float deltadx = dx / NumSamples;
    float deltady = dy / NumSamples;
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[0][0] = cos(deltadegree * i) * (1.0f + i * deltascale);
      Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
      Hmatrix[i].Hmatrix[0][2] = i * deltadx;
      Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
      Hmatrix[i].Hmatrix[1][1] = cos(deltadegree * i) * (1.0f + i * deltascale);
      Hmatrix[i].Hmatrix[1][2] = i * deltady;
      Hmatrix[i].Hmatrix[2][0] = i * deltapx;
      Hmatrix[i].Hmatrix[2][1] = i * deltapy;
      Hmatrix[i].Hmatrix[2][2] = 1;

      Homography::MatrixInverse(Hmatrix[i].Hmatrix, IHmatrix[i].Hmatrix);
    }
  }

  ////////////////////////////////////
  // These functions are used to generate the Projective Motion Blur Images
  ////////////////////////////////////
  // i: postive forward, negative backward
  void WarpImageGray(float* InputImg, float* inputWeight, int iwidth,
                     int iheight, float* OutputImg, float* outputWeight,
                     int width, int height, int i);
  void WarpImageRgb(float* InputImgR, float* InputImgG, float* InputImgB,
                    float* inputWeight, int iwidth, int iheight,
                    float* OutputImgR, float* OutputImgG, float* OutputImgB,
                    float* outputWeight, int width, int height, int i);
  // bforward: true forward, false backward
  void GenerateMotionBlurImgGray(float* InputImg, float* inputWeight,
                                 int iwidth, int iheight, float* BlurImg,
                                 float* outputWeight, int width, int height,
                                 bool bforward = true);
  void GenerateMotionBlurImgRgb(float* InputImgR, float* InputImgG,
                                float* InputImgB, float* inputWeight,
                                int iwidth, int iheight, float* BlurImgR,
                                float* BlurImgG, float* BlurImgB,
                                float* outputWeight, int width, int height,
                                bool bforward = true);

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

  // Multiscale version: not very useful...
  void ProjectiveMotionRLDeblurMultiScaleGray(float* BlurImg, int iwidth,
                                              int iheight, float* DeblurImg,
                                              int width, int height,
                                              int Niter = 10, int Nscale = 5,
                                              bool bPoisson = true);

  // These are the homography sequence for Projective motion blur model
  Homography Hmatrix[NumSamples];
  Homography IHmatrix[NumSamples];

 private:
  float getSpsWeight(float aValue) const;

  // These are buffer and lookup table variables
  float BilateralTable[256];
  float SpsTable[256];
  std::vector<float> mInitialInputWeight;
  std::vector<float> WarpImgBuffer;
  std::vector<float> WarpImgBufferR;
  std::vector<float> WarpImgBufferG;
  std::vector<float> WarpImgBufferB;
  std::vector<float> WarpWeightBuffer;
  std::vector<float> BlurImgBuffer;
  std::vector<float> BlurImgBufferR;
  std::vector<float> BlurImgBufferG;
  std::vector<float> BlurImgBufferB;
  std::vector<float> BlurWeightBuffer;
  std::vector<float> ErrorImgBuffer;
  std::vector<float> ErrorImgBufferR;
  std::vector<float> ErrorImgBufferG;
  std::vector<float> ErrorImgBufferB;
  std::vector<float> ErrorWeightBuffer;

  // This variables are used for RMS computation
  std::vector<float> GroundTruthImg;
  std::vector<float> GroundTruthImgR;
  std::vector<float> GroundTruthImgG;
  std::vector<float> GroundTruthImgB;

  // Random values generation
  std::random_device mRandomDevice;
  std::mt19937 mRandomEngine{mRandomDevice()};
};
