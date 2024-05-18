#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "BlurKernelGenerator.hpp"
#include "BlurUtils.hpp"
#include "DeblurParameters.hpp"
#include "EmptyErrorCalculator.hpp"
#include "EmptyRegularizer.hpp"
#include "GaussianNoiseGenerator.hpp"
#include "ImResize.h"
#include "MotionBlurMaker.hpp"
#include "RLDeblurrer.hpp"
#include "RMSErrorCalculator.hpp"
#include "bitmap.h"

constexpr auto fileExtension = ".bmp";

class BoxBlurImageGenerator : public IBlurImageGenerator {
 public:
  ~BoxBlurImageGenerator() override = default;

  static constexpr int kBoxX = 4;
  static constexpr int kBoxY = 4;
  static constexpr float kBoxWeight = 1.0f / (kBoxX * kBoxY);

  // bforward: true forward, false backward
  void blurGray(float* InputImg, float* inputWeight, int iwidth, int iheight,
                float* BlurImg, float* outputWeight, int width, int height,
                bool bforward) override {
    const float blurDirection = bforward ? 1.0f : -1.0f;

    for (int y = 0, index = 0; y < height; y++) {
      for (int x = 0; x < width; x++, index++) {
        // TODO: Set x and y span
        float weightSum = 0;
        float blurSum = 0;
        for (int fy = y; fy < y + kBoxX * blurDirection; fy++) {
          for (int fx = x; fx < x + kBoxY * blurDirection; fx++) {
            if (inputWeight && fx >= 0 && fx < iwidth - 1 && fy >= 0 &&
                fy < iheight - 1) {
              weightSum += inputWeight[fx + fy * iwidth] * kBoxWeight;
            }

            // TODO: Set border conditions
            if (fx < 0) continue;
            if (fy < 0) continue;
            if (fx >= iwidth) continue;
            if (fy >= iheight) continue;

            blurSum += InputImg[fx + fy * iwidth] * kBoxWeight;
          }
        }

        if (inputWeight) {
          outputWeight[index] = weightSum;
        } else {
          outputWeight[index] = 1.0f;
        }

        BlurImg[index] = blurSum;
      }
    }
  }

  void blurRgb(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* BlurImgR,
               float* BlurImgG, float* BlurImgB, float* outputWeight, int width,
               int height, bool bforward) override {
    const float blurDirection = bforward ? 1.0f : -1.0f;

    for (int y = 0, index = 0; y < height; y++) {
      for (int x = 0; x < width; x++, index++) {
        // TODO: Set x and y span
        float weightSum = 0;
        float blurSumR = 0;
        float blurSumG = 0;
        float blurSumB = 0;
        for (int fy = y; fy < y + kBoxX * blurDirection; fy++) {
          for (int fx = x; fx < x + kBoxY * blurDirection; fx++) {
            if (inputWeight && fx >= 0 && fx < iwidth - 1 && fy >= 0 &&
                fy < iheight - 1) {
              weightSum += inputWeight[fx + fy * iwidth] * kBoxWeight;
            }

            // TODO: Set border conditions
            if (fx < 0) continue;
            if (fy < 0) continue;
            if (fx >= iwidth) continue;
            if (fy >= iheight) continue;

            blurSumR += InputImgR[fx + fy * iwidth] * kBoxWeight;
            blurSumG += InputImgG[fx + fy * iwidth] * kBoxWeight;
            blurSumB += InputImgB[fx + fy * iwidth] * kBoxWeight;
          }
        }

        if (inputWeight) {
          outputWeight[index] = weightSum;
        } else {
          outputWeight[index] = 1.0f;
        }

        BlurImgR[index] = blurSumR;
        BlurImgG[index] = blurSumG;
        BlurImgB[index] = blurSumB;
      }
    }
  }

  void SetBuffer(int width, int height) override {}

  void ClearBuffer() override {}
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s image_filename [blur_type]\n", argv[0]);
    return EXIT_SUCCESS;
  }

  std::string fname{argv[1]};
  const auto pos = fname.find(fileExtension);
  if (pos == std::string::npos) {
    printf("Expected %s to end with %s\n", fname.c_str(), fileExtension);
    return EXIT_SUCCESS;
  }
  const auto prefix = fname.substr(0, pos);

  int width = 0, height = 0;
  std::vector<float> fImg[3];

  printf("Load Image: %s\n", fname.c_str());
  readBMPchannels(fname, fImg[0], fImg[1], fImg[2], width, height);
  int blurwidth = width, blurheight = height;

  if (fImg[0].empty()) {
    printf("Error reading %s\n", fname.c_str());
    return EXIT_SUCCESS;
  }

  std::vector<float> bImg[3];
  std::vector<float> deblurImg[3];
  std::vector<float> inputWeight;
  std::vector<float> outputWeight(width * height);
  float RMSError = NAN;
  bImg[0].resize(width * height);
  bImg[1].resize(width * height);
  bImg[2].resize(width * height);
  deblurImg[0].resize(width * height);
  deblurImg[1].resize(width * height);
  deblurImg[2].resize(width * height);

  ///////////////////////////////////
  printf("Set Blur Kernel Parameters\n");
  constexpr int kernelHalfWidth = 10;
  constexpr int kernelHalfHeight = 10;
  BlurKernelGenerator blurGenerator{kernelHalfWidth,
                                    kernelHalfHeight,
                                    BlurKernelGenerator::Border::ISOLATED,
                                    fImg[0].data(),
                                    width,
                                    height};
  RMSErrorCalculator errorCalculator;
  EmptyErrorCalculator emptyErrorCalculator;

  BoxBlurImageGenerator sampleGenerator;

  ///////////////////////////////////
  // Ground truth blur kernel setup
  std::vector<float> kernelImg[3];
  kernelImg[0].resize(width * height);
  kernelImg[1].resize(width * height);
  kernelImg[2].resize(width * height);
  memset(kernelImg[0].data(), 0, width * height * sizeof(float));
  memset(kernelImg[1].data(), 0, width * height * sizeof(float));
  memset(kernelImg[2].data(), 0, width * height * sizeof(float));
  // kernelImg[0][width/2 +  width * height/2] = 1.0f;
  // kernelImg[1][width/2 + width * height/2] = 1.0f;
  // kernelImg[2][width/2 + width * height/2] = 1.0f;

  // Line x+
  kernelImg[0][0] = 0.25f;
  kernelImg[1][0] = 0.25f;
  kernelImg[2][0] = 0.25f;
  kernelImg[0][1] = 0.25f;
  kernelImg[1][1] = 0.25f;
  kernelImg[2][1] = 0.25f;
  kernelImg[0][2] = 0.25f;
  kernelImg[1][2] = 0.25f;
  kernelImg[2][2] = 0.25f;
  kernelImg[0][3] = 0.25f;
  kernelImg[1][3] = 0.25f;
  kernelImg[2][3] = 0.25f;

  // Line x-
  // kernelImg[0][0] = 0.25f;
  // kernelImg[1][0] = 0.25f;
  // kernelImg[2][0] = 0.25f;
  // kernelImg[0][width-1] = 0.25f;
  // kernelImg[1][width-1] = 0.25f;
  // kernelImg[2][width-1] = 0.25f;
  // kernelImg[0][width-2] = 0.25f;
  // kernelImg[1][width-2] = 0.25f;
  // kernelImg[2][width-2] = 0.25f;
  // kernelImg[0][width-3] = 0.25f;
  // kernelImg[1][width-3] = 0.25f;
  // kernelImg[2][width-3] = 0.25f;

  // Line y+
  // kernelImg[0][0] = 0.25f;
  // kernelImg[1][0] = 0.25f;
  // kernelImg[2][0] = 0.25f;
  // kernelImg[0][width] = 0.25f;
  // kernelImg[1][width] = 0.25f;
  // kernelImg[2][width] = 0.25f;
  // kernelImg[0][2*width] = 0.25f;
  // kernelImg[1][2*width] = 0.25f;
  // kernelImg[2][2*width] = 0.25f;
  // kernelImg[0][3*width] = 0.25f;
  // kernelImg[1][3*width] = 0.25f;
  // kernelImg[2][3*width] = 0.25f;

  // Line y-
  // kernelImg[0][0] = 0.25f;
  // kernelImg[1][0] = 0.25f;
  // kernelImg[2][0] = 0.25f;
  // kernelImg[0][(height-1)*width] = 0.25f;
  // kernelImg[1][(height-1)*width] = 0.25f;
  // kernelImg[2][(height-1)*width] = 0.25f;
  // kernelImg[0][(height-2)*width] = 0.25f;
  // kernelImg[1][(height-2)*width] = 0.25f;
  // kernelImg[2][(height-2)*width] = 0.25f;
  // kernelImg[0][(height-3)*width] = 0.25f;
  // kernelImg[1][(height-3)*width] = 0.25f;
  // kernelImg[2][(height-3)*width] = 0.25f;

  // kernelImg[0][0] = 1.0f;
  // kernelImg[1][0] = 1.0f;
  // kernelImg[2][0] = 1.0f;

  generateMotionBlurredImage(kernelImg, inputWeight, outputWeight, width,
                             height, blurwidth, blurheight, prefix,
                             blurGenerator, errorCalculator, deblurImg);

  // errorCalculator.SetGroundTruthImgRgb(
  //     deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
  //     height);  // This is for error computation

  // Scale up kernel
  // std::for_each(std::begin(deblurImg[0]), std::end(deblurImg[0]),
  //               [](auto& aValue) { aValue *= 200; });
  // std::for_each(std::begin(deblurImg[1]), std::end(deblurImg[1]),
  //               [](auto& aValue) { aValue *= 200; });
  // std::for_each(std::begin(deblurImg[2]), std::end(deblurImg[2]),
  //               [](auto& aValue) { aValue *= 200; });

  writeBMPchannels("ground_truth_kernel", width, height, deblurImg[0],
                   deblurImg[1], deblurImg[2]);

  ///////////////////////////////////
  generateMotionBlurredImage(fImg, inputWeight, outputWeight, width, height,
                             blurwidth, blurheight, prefix, sampleGenerator,
                             emptyErrorCalculator, bImg);

  // Add noise
  // const float sigma = 2.0f;
  // const std::string noisePrefix =
  //     prefix + "_blur_noise_sigma" + std::to_string(sigma) + "_";
  // GaussianNoiseGenerator noiseGenerator(sigma);
  // addNoiseToImage(bImg, width, height, blurwidth, blurheight, noisePrefix,
  //                 noiseGenerator, errorCalculator);

  ///////////////////////////////////
  // Main Deblurring algorithm
  EmptyRegularizer emptyRegularizer;
  RLDeblurrer rLDeblurrer{sampleGenerator, emptyErrorCalculator};

  printf("Initial Estimation is a line kernel\n");
  memset(deblurImg[0].data(), 0, width * height * sizeof(float));
  memset(deblurImg[1].data(), 0, width * height * sizeof(float));
  memset(deblurImg[2].data(), 0, width * height * sizeof(float));

  deblurImg[0][0] = 0.25f;
  deblurImg[0][0] = 0.25f;
  deblurImg[0][0] = 0.25f;
  deblurImg[0][0] = 0.25f;
  deblurImg[1][0] = 0.25f;
  deblurImg[1][0] = 0.25f;
  deblurImg[1][0] = 0.25f;
  deblurImg[1][0] = 0.25f;
  deblurImg[2][0] = 0.25f;
  deblurImg[2][0] = 0.25f;
  deblurImg[2][0] = 0.25f;
  deblurImg[2][0] = 0.25f;

  // Levin et. al. Siggraph07's matlab implementation also take around 400
  // iterations Sadly, the algorithm needs such a lot of iterations to produce
  // good results Most running time were spent on bicubic interpolation, it
  // would be much faster if this step was implemented in GPU...

  // Load Initial Guess, if you have...
  //   readBMP("", deblurImg[0], deblurImg[1], deblurImg[2], width, height);

  printf("Basic Algorithm:\n");

  DeblurParameters rLParams{.Niter = 500, .bPoisson = true};
  // rLDeblurrer.deblurRgb(bImg[0].data(), bImg[1].data(), bImg[2].data(),
  //                       blurwidth, blurheight, deblurImg[0].data(),
  //                       deblurImg[1].data(), deblurImg[2].data(), width,
  //                       height, rLParams, emptyRegularizer, 0.0);

  // Gray deblur
  rLDeblurrer.deblurGray(bImg[0].data(), blurwidth, blurheight,
                         deblurImg[0].data(), width, height, rLParams,
                         emptyRegularizer, 0.0);

  rLDeblurrer.deblurGray(bImg[1].data(), blurwidth, blurheight,
                         deblurImg[1].data(), width, height, rLParams,
                         emptyRegularizer, 0.0);

  rLDeblurrer.deblurGray(bImg[2].data(), blurwidth, blurheight,
                         deblurImg[2].data(), width, height, rLParams,
                         emptyRegularizer, 0.0);

  RMSError = errorCalculator.calculateErrorRgb(
      deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
      height);

  fname = prefix + "_blurKernel_" + std::to_string(RMSError * 255.0f) +
          fileExtension;
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);

  // Scale up kernel
  std::for_each(std::begin(deblurImg[0]), std::end(deblurImg[0]),
                [](auto& aValue) { aValue *= 100; });
  std::for_each(std::begin(deblurImg[1]), std::end(deblurImg[1]),
                [](auto& aValue) { aValue *= 100; });
  std::for_each(std::begin(deblurImg[2]), std::end(deblurImg[2]),
                [](auto& aValue) { aValue *= 100; });

  writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                   deblurImg[2]);

  return EXIT_SUCCESS;
}
