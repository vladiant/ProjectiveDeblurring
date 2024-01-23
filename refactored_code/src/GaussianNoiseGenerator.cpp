#include <GaussianNoiseGenerator.h>

GaussianNoiseGenerator::GaussianNoiseGenerator() { mRandomEngine.seed(kSeed); }

float GaussianNoiseGenerator::normalrand() {
  std::normal_distribution<float> normalDist(0.0f, 1.0f);
  // float val = 0;
  // for (int i = 0; i != 12; ++i) val += ((float)(rand()) / RAND_MAX);
  // return val - 6.0f;
  return normalDist(mRandomEngine);
}

// Noise variance = amp
void GaussianNoiseGenerator::addNoiseGray(float* Img, int width, int height,
                                          float amp) {
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