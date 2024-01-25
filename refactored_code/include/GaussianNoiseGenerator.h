#pragma once

#include <random>

#include "INoiseGenerator.h"

class GaussianNoiseGenerator : public INoiseGenerator {
 public:
  // Random engine seed
  constexpr static int kSeed = 1234;

  // Noise variance = amp
  GaussianNoiseGenerator(float aSigma);

  ////////////////////////////////////
  // These functions are used to generate noise
  ////////////////////////////////////
  void addNoiseGray(float* Img, int width, int height, float* aOutImg) override;

  void addNoiseRgb(float* ImgR, float* ImgG, float* ImgB, int width, int height,
                   float* aOutImgR, float* aOutImgG, float* aOutImgB) override;

 private:
  // Random values generation
  std::random_device mRandomDevice;
  std::mt19937 mRandomEngine{mRandomDevice()};

  // Normal random number generator, variance = 1
  float normalrand();

  // Noise parameters
  float mSigma;
};
