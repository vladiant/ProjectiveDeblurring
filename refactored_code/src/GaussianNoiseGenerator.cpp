#include <GaussianNoiseGenerator.h>

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
  int x, y, index;
  float random, noise;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      random = normalrand() / 255.0f;
      noise = mSigma * random;
      aOutImg[index] = Img[index] + noise;
      if (aOutImg[index] > 1.0f) aOutImg[index] = 1.0f;
      if (aOutImg[index] < 0.0f) aOutImg[index] = 0.0f;
    }
  }
}