#include "GaussianNoiseGenerator.hpp"

#include <algorithm>

GaussianNoiseGenerator::GaussianNoiseGenerator(float aSigma) : mSigma(aSigma) {
  mRandomEngine.seed(kSeed);
}

float GaussianNoiseGenerator::normalrand() {
  std::normal_distribution<float> normalDist(0.0f, 1.0f);
  // float val = 0;
  // for (int i = 0; i != 12; ++i) val += ((float)(rand()) / RAND_MAX);
  // return val - 6.0f;
  return normalDist(mRandomEngine);
}

void GaussianNoiseGenerator::addNoiseGray(float* Img, int width, int height,
                                          float* aOutImg) {
  for (int y = 0, index = 0; y < height; y++) {
    for (int x = 0; x < width; x++, index++) {
      const float random = normalrand() / 255.0f;
      const float noise = mSigma * random;
      aOutImg[index] = Img[index] + noise;
      aOutImg[index] = std::clamp(aOutImg[index], 0.0f, 1.0f);
    }
  }
}

void GaussianNoiseGenerator::addNoiseRgb(float* ImgR, float* ImgG, float* ImgB,
                                         int width, int height, float* aOutImgR,
                                         float* aOutImgG, float* aOutImgB) {
  for (int y = 0, index = 0; y < height; y++) {
    for (int x = 0; x < width; x++, index++) {
      const float random = normalrand() / 255.0f;
      const float noise = mSigma * random;
      aOutImgR[index] = ImgR[index] + noise;
      aOutImgG[index] = ImgG[index] + noise;
      aOutImgB[index] = ImgB[index] + noise;
      aOutImgR[index] = std::clamp(aOutImgR[index], 0.0f, 1.0f);
      aOutImgG[index] = std::clamp(aOutImgG[index], 0.0f, 1.0f);
      aOutImgB[index] = std::clamp(aOutImgB[index], 0.0f, 1.0f);
    }
  }
}