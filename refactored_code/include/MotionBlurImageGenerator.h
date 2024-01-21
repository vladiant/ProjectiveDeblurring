#pragma once

#include <vector>

#include "IBlurImageGenerator.h"
#include "homography.h"

class MotionBlurImageGenerator : public IBlurImageGenerator {
 public:
  constexpr static int NumSamples = 30;

  MotionBlurImageGenerator();

  // bforward: true forward, false backward
  void blurGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                float* BlurImg, float* outputWeight, int width, int height,
                bool bforward) override;
  void blurRgb(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* BlurImgR,
               float* BlurImgG, float* BlurImgB, float* outputWeight, int width,
               int height, bool bforward) override;

  void SetBuffer(int width, int height) override;
  void ClearBuffer() override;

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

  // These are the homography sequence for Projective motion blur model
  Homography Hmatrix[NumSamples];
  Homography IHmatrix[NumSamples];

 private:
  std::vector<float> mWarpImgBuffer;
  std::vector<float> mWarpImgBufferR;
  std::vector<float> mWarpImgBufferG;
  std::vector<float> mWarpImgBufferB;
  std::vector<float> mWarpWeightBuffer;

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
};