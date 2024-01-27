#include "RMSErrorCalculator.h"

#include <cmath>
#include <cstdio>
#include <cstring>

RMSErrorCalculator::~RMSErrorCalculator() {
  mGroundTruthImg.clear();
  mGroundTruthImgR.clear();
  mGroundTruthImgG.clear();
  mGroundTruthImgB.clear();
}

void RMSErrorCalculator::SetGroundTruthImgGray(float* GroundTruth, int width,
                                               int height) {
  mGroundTruthImg.resize(width * height);
  memcpy(mGroundTruthImg.data(), GroundTruth, width * height * sizeof(float));
}

void RMSErrorCalculator::SetGroundTruthImgRgb(float* GroundTruthR,
                                              float* GroundTruthG,
                                              float* GroundTruthB, int width,
                                              int height) {
  mGroundTruthImgR.resize(width * height);
  mGroundTruthImgG.resize(width * height);
  mGroundTruthImgB.resize(width * height);
  memcpy(mGroundTruthImgR.data(), GroundTruthR, width * height * sizeof(float));
  memcpy(mGroundTruthImgG.data(), GroundTruthG, width * height * sizeof(float));
  memcpy(mGroundTruthImgB.data(), GroundTruthB, width * height * sizeof(float));
}

float RMSErrorCalculator::calculateErrorRgb(float* ImgR, float* ImgG,
                                            float* ImgB, int width,
                                            int height) {
  const float rmseErrorR =
      ComputeRMSErrorGray(mGroundTruthImgR.data(), ImgR, width, height);
  const float rmseErrorG =
      ComputeRMSErrorGray(mGroundTruthImgG.data(), ImgG, width, height);
  const float rmseErrorB =
      ComputeRMSErrorGray(mGroundTruthImgB.data(), ImgB, width, height);
  printf("RMS Error R: %f G: %f B: %f\n", rmseErrorR, rmseErrorG, rmseErrorB);
  return (rmseErrorR + rmseErrorG + rmseErrorB) / 3.0f;
}

float RMSErrorCalculator::calculateErrorGray(float* Img, int width,
                                             int height) {
  const float rmseError =
      ComputeRMSErrorGray(mGroundTruthImg.data(), Img, width, height);
  printf("RMS Error: %f\n", rmseError);
  return rmseError;
}

float RMSErrorCalculator::ComputeRMSErrorGray(float* GroundTruth,
                                              float* DeblurredImg, int width,
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