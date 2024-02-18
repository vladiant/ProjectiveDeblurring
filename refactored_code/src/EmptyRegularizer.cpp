#include "EmptyRegularizer.hpp"

void EmptyRegularizer::applyRegularizationGray(
    [[maybe_unused]] float* DeblurImg, [[maybe_unused]] int width,
    [[maybe_unused]] int height, [[maybe_unused]] bool bPoisson,
    [[maybe_unused]] float lambda) {}

void EmptyRegularizer::applyRegularizationRgb(
    [[maybe_unused]] float* DeblurImgR, [[maybe_unused]] float* DeblurImgG,
    [[maybe_unused]] float* DeblurImgB, [[maybe_unused]] int width,
    [[maybe_unused]] int height, [[maybe_unused]] bool bPoisson,
    [[maybe_unused]] float lambda) {}