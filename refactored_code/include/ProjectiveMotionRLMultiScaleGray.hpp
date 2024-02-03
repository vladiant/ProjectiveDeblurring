#pragma once

#include <vector>

#include "Homography.hpp"

// Multiscale version: not very useful...
class ProjectiveMotionRLMultiScaleGray {
 public:
  constexpr static int NumSamples = 30;

  ProjectiveMotionRLMultiScaleGray();
  ~ProjectiveMotionRLMultiScaleGray();

  void ProjectiveMotionRLDeblurMultiScaleGray(float* BlurImg, int iwidth,
                                              int iheight, float* DeblurImg,
                                              int width, int height,
                                              int Niter = 10, int Nscale = 5,
                                              bool bPoisson = true);

  // These are the homography sequence for Projective motion blur model
  Homography Hmatrix[NumSamples]{};
  Homography IHmatrix[NumSamples]{};

 private:
  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height) {
    mWarpImgBuffer.resize(width * height);
    mWarpWeightBuffer.resize(width * height);
    mBlurImgBuffer.resize(width * height);
    mBlurWeightBuffer.resize(width * height);
    mErrorImgBuffer.resize(width * height);
    mErrorWeightBuffer.resize(width * height);
  }

  void ClearBuffer() {
    mWarpImgBuffer.clear();
    mWarpWeightBuffer.clear();
    mBlurImgBuffer.clear();
    mBlurWeightBuffer.clear();
    mErrorImgBuffer.clear();
    mErrorWeightBuffer.clear();
  }

  ////////////////////////////////////
  // These functions are used to generate the Projective Motion Blur Images
  ////////////////////////////////////
  // i: postive forward, negative backward
  void WarpImageGray(float* InputImg, float* inputWeight, int iwidth,
                     int iheight, float* OutputImg, float* outputWeight,
                     int width, int height, int i);
  // bforward: true forward, false backward
  void GenerateMotionBlurImgGray(float* InputImg, float* inputWeight,
                                 int iwidth, int iheight, float* BlurImg,
                                 float* outputWeight, int width, int height,
                                 bool bforward = true);

  std::vector<float> mBlurImgBuffer;
  std::vector<float> mBlurWeightBuffer;
  std::vector<float> mErrorImgBuffer;
  std::vector<float> mErrorWeightBuffer;
  std::vector<float> mWarpImgBuffer;
  std::vector<float> mWarpWeightBuffer;
};