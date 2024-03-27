#include "BlurUtils.hpp"

#include <cstdio>

#include "bitmap.h"

static constexpr auto fileExtension = ".bmp";

void generateMotionBlurredImage(std::vector<float> (&aInitialImage)[3],
                                std::vector<float>& aInputWeight,
                                std::vector<float>& aOutputWeight, int aWidth,
                                int aHeight, int aBlurWidth, int aBlurHeight,
                                const std::string& aPrefix,
                                IBlurImageGenerator& aBlurGenerator,
                                IErrorCalculator& aErrorCalculator,
                                std::vector<float> (&aBlurredImage)[3]) {
  printf("Generate Motion Blurred Image\n");
  aBlurGenerator.blurGray(aInitialImage[0].data(), aInputWeight.data(), aWidth,
                          aHeight, aBlurredImage[0].data(),
                          aOutputWeight.data(), aBlurWidth, aBlurHeight, true);
  aBlurGenerator.blurGray(aInitialImage[1].data(), aInputWeight.data(), aWidth,
                          aHeight, aBlurredImage[1].data(),
                          aOutputWeight.data(), aBlurWidth, aBlurHeight, true);
  aBlurGenerator.blurGray(aInitialImage[2].data(), aInputWeight.data(), aWidth,
                          aHeight, aBlurredImage[2].data(),
                          aOutputWeight.data(), aBlurWidth, aBlurHeight, true);

  if (!aPrefix.empty()) {
    const float RMSError = aErrorCalculator.calculateErrorRgb(
        aBlurredImage[0].data(), aBlurredImage[1].data(),
        aBlurredImage[2].data(), aWidth, aHeight);
    const std::string fname =
        aPrefix + "_blur_" + std::to_string(RMSError * 255.0f) + fileExtension;
    printf("Save Blurred Image to: %s\n", fname.c_str());
    writeBMPchannels(fname, aBlurWidth, aBlurHeight, aBlurredImage[0],
                     aBlurredImage[1], aBlurredImage[2]);
  }
}

void addNoiseToImage(std::vector<float> (&bImg)[3], int width, int height,
                     int blurWidth, int blurHeight, const std::string& prefix,
                     INoiseGenerator& noiseGenerator,
                     IErrorCalculator& errorCalculator) {
  noiseGenerator.addNoiseGray(bImg[0].data(), width, height, bImg[0].data());
  noiseGenerator.addNoiseGray(bImg[1].data(), width, height, bImg[1].data());
  noiseGenerator.addNoiseGray(bImg[2].data(), width, height, bImg[2].data());

  if (!prefix.empty()) {
    const float RMSError = errorCalculator.calculateErrorRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), width, height);
    const std::string fname =
        prefix + std::to_string(RMSError * 255.0f) + fileExtension;
    printf("Save Blurred Image to: %s\n", fname.c_str());
    writeBMPchannels(fname, blurWidth, blurHeight, bImg[0], bImg[1], bImg[2]);
  }
}