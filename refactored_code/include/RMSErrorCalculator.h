#pragma once

#include <vector>

#include "IErrorCalculator.h"

class RMSErrorCalculator : public IErrorCalculator {
 public:
  ~RMSErrorCalculator() override;

  void SetGroundTruthImgGray(float* GroundTruth, int width, int height);

  void SetGroundTruthImgRgb(float* GroundTruthR, float* GroundTruthG,
                            float* GroundTruthB, int width, int height);

  ////////////////////////////////////
  // These functions are used to compute Errors
  ////////////////////////////////////
  float calculateErrorRgb(float* ImgR, float* ImgG, float* ImgB, int width,
                          int height) override;

  float calculateErrorGray(float* Img, int width, int height) override;

  float ComputeRMSErrorGray(float* GroundTruth, float* DeblurredImg, int width,
                            int height);

 private:
  // This variables are used for RMS computation
  std::vector<float> mGroundTruthImg;
  std::vector<float> mGroundTruthImgR;
  std::vector<float> mGroundTruthImgG;
  std::vector<float> mGroundTruthImgB;
};
