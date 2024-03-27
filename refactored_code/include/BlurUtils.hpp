#pragma once

#include <string>
#include <vector>

#include "IBlurImageGenerator.hpp"
#include "IErrorCalculator.hpp"
#include "INoiseGenerator.hpp"

void generateMotionBlurredImage(std::vector<float> (&aInitialImage)[3],
                                std::vector<float>& aInputWeight,
                                std::vector<float>& aOutputWeight, int aWidth,
                                int aHeight, int aBlurWidth, int aBlurHeight,
                                const std::string& aPrefix,
                                IBlurImageGenerator& aBlurGenerator,
                                IErrorCalculator& aErrorCalculator,
                                std::vector<float> (&aBlurredImage)[3]);

void addNoiseToImage(std::vector<float> (&bImg)[3], int width, int height,
                     int blurWidth, int blurHeight, const std::string& prefix,
                     INoiseGenerator& noiseGenerator,
                     IErrorCalculator& errorCalculator);