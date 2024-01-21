#pragma once

#include <vector>

#include "homography.h"

class MotionBlurImageGenerator {
 public:
  constexpr static int NumSamples = 30;

  MotionBlurImageGenerator();

  ////////////////////////////////////
  // These functions are used to set the homography
  ////////////////////////////////////
  void SetHomography(Homography H, int i);

  void SetGlobalRotation(float degree);
  void SetGlobalScaling(float scalefactor);
  void SetGlobalTranslation(float dx, float dy);
  void SetGlobalPerspective(float px, float py);
  void SetGlobalParameters(float degree, float scalefactor, float px, float py,
                           float dx, float dy);

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

  // These are the homography sequence for Projective motion blur model
  Homography Hmatrix[NumSamples];
  Homography IHmatrix[NumSamples];

  void SetBuffer(int width, int height);
  void ClearBuffer();

 private:
  std::vector<float> mWarpImgBuffer;
  std::vector<float> mWarpImgBufferR;
  std::vector<float> mWarpImgBufferG;
  std::vector<float> mWarpImgBufferB;
  std::vector<float> mWarpWeightBuffer;
};