#pragma once

#include <random>

class GaussianNoiseGenerator {
 public:
  // Random engine seed
  constexpr static int kSeed = 1234;

  // Noise variance = amp
  GaussianNoiseGenerator(float aSigma);

  ////////////////////////////////////
  // These functions are used to generate noise
  ////////////////////////////////////
  // Normal random number generator, variance = 1
  float normalrand();

  void addNoiseGray(float* Img, int width, int height);

 private:
  // Random values generation
  std::random_device mRandomDevice;
  std::mt19937 mRandomEngine{mRandomDevice()};

  // Noise parameters
  float mSigma;
};
