#include "KernelRegularizer.hpp"

KernelRegularizer::KernelRegularizer() {}

void KernelRegularizer::SetBuffer(int width, int height) {}

void KernelRegularizer::ClearBuffer() {}

void KernelRegularizer::applyRegularizationGray(float* DeblurImg, int width,
                                                int height, bool bPoisson,
                                                float lambda) {
  // Sum of abs weight should be one
  // There can not be negative weight
}

void KernelRegularizer::applyRegularizationRgb(float* DeblurImgR,
                                               float* DeblurImgG,
                                               float* DeblurImgB, int width,
                                               int height, bool bPoisson,
                                               float lambda) {}