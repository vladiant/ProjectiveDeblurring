#include "KernelRegularizer.hpp"

KernelRegularizer::KernelRegularizer() {}

void KernelRegularizer::SetBuffer(int width, int height) {}

void KernelRegularizer::ClearBuffer() {}

void KernelRegularizer::applyRegularizationGray(float* DeblurImg, int width,
                                                int height, bool bPoisson,
                                                float lambda) {
  // Sum of abs weight should be one
  // There can not be negative weight
  float sum = 0;
  for (int index = 0; index < width * height; index++) {
    if (DeblurImg[index] > 0) {
      sum += DeblurImg[index];
    } else {
      DeblurImg[index] = 0;
    }
  }

  // TODO: fix comparison to zero
  if (sum == 0) {
    return;
  }

  for (int index = 0; index < width * height; index++) {
    DeblurImg[index] /= sum;
  }
}

void KernelRegularizer::applyRegularizationRgb(float* DeblurImgR,
                                               float* DeblurImgG,
                                               float* DeblurImgB, int width,
                                               int height, bool bPoisson,
                                               float lambda) {}