#include "EmptyErrorCalculator.h"

#include <cstdio>

float EmptyErrorCalculator::calculateErrorRgb([[maybe_unused]] float* ImgR,
                                              [[maybe_unused]] float* ImgG,
                                              [[maybe_unused]] float* ImgB,
                                              [[maybe_unused]] int width,
                                              [[maybe_unused]] int height) {
  printf(".");
  return 0;
}

float EmptyErrorCalculator::calculateErrorGray([[maybe_unused]] float* Img,
                                               [[maybe_unused]] int width,
                                               [[maybe_unused]] int height) {
  printf(".");
  return 0;
}