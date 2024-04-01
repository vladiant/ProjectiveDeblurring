#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "BlurUtils.hpp"
#include "DeblurParameters.hpp"
#include "EmptyErrorCalculator.hpp"
#include "EmptyRegularizer.hpp"
#include "GaussianNoiseGenerator.hpp"
#include "ImResize.h"
#include "LaplacianRegularizer.hpp"
#include "MotionBlurImageGenerator.hpp"
#include "MotionBlurMaker.hpp"
#include "ProjectiveMotionRLMultiScaleGray.hpp"
#include "RLDeblurrer.hpp"
#include "RMSErrorCalculator.hpp"
#include "bitmap.h"

constexpr auto fileExtension = ".bmp";

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
  printf("Set Projective Model Parameter\n");
  MotionBlurImageGenerator blurGenerator;
  RMSErrorCalculator errorCalculator;
  EmptyErrorCalculator emptyErrorCalculator;

  int blurType = 0;
  if (argc > 2) {
    const std::string blurTypeArg{argv[2]};
    const auto convResult = std::from_chars(
        blurTypeArg.data(), blurTypeArg.data() + blurTypeArg.size(), blurType);
    if (convResult.ec != std::errc()) {
      printf("Error convering %s to int\n", blurTypeArg.c_str());
      return EXIT_SUCCESS;
    }
  }

  if (!setBlur(blurType, blurGenerator)) {
    return EXIT_SUCCESS;
  }

  ///////////////////////////////////
  errorCalculator.SetGroundTruthImgRgb(
      fImg[0].data(), fImg[1].data(), fImg[2].data(), width,
      height);  // This is for error computation

  ///////////////////////////////////
  generateMotionBlurredImage(fImg, inputWeight, outputWeight, width, height,
                             blurwidth, blurheight, prefix, blurGenerator,
                             errorCalculator, bImg);

  // Add noise
  const float sigma = 2.0f;
  const std::string noisePrefix =
      prefix + "_blur_noise_sigma" + std::to_string(sigma) + "_";
  GaussianNoiseGenerator noiseGenerator(sigma);
  addNoiseToImage(bImg, width, height, blurwidth, blurheight, noisePrefix,
                  noiseGenerator, errorCalculator);

  ///////////////////////////////////
  EmptyRegularizer emptyRegularizer;

  {
    printf("Initial Estimation is the blur image\n");
    ImChoppingGray(bImg[0].data(), blurwidth, blurheight, deblurImg[0].data(),
                   width, height);
    ImChoppingGray(bImg[1].data(), blurwidth, blurheight, deblurImg[1].data(),
                   width, height);
    ImChoppingGray(bImg[2].data(), blurwidth, blurheight, deblurImg[2].data(),
                   width, height);

    printf("Laplacian Regularization Algorithm:\n");

    LaplacianRegularizer laplRegularizer;
    DeblurParameters laplRLParams{.Niter = 100, .bPoisson = true};
    RLDeblurrer rLDeblurrerLaplReg{blurGenerator, emptyErrorCalculator};

    //   rLDeblurrerLaplReg.deblurRgb(
    //       bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth,
    //       blurheight, deblurImg[0].data(), deblurImg[1].data(),
    //       deblurImg[2].data(), width, height, DeblurParameters{.Niter = 100,
    //     .bPoisson = true}, laplRegularizer, 0.5f);
    rLDeblurrerLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, laplRLParams, laplRegularizer, 1.0f);
    rLDeblurrerLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, laplRLParams, laplRegularizer, 0.5f);
    rLDeblurrerLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, laplRLParams, laplRegularizer, 0.25f);
    rLDeblurrerLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, laplRLParams, laplRegularizer, 0.125f);

    DeblurParameters rLParams{.Niter = 100, .bPoisson = true};
    rLDeblurrerLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, rLParams, emptyRegularizer, 0.0);
    RMSError = errorCalculator.calculateErrorRgb(
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height);
    //   sprintf(fname, "%s_deblurSpsReg_%f.bmp", prefix, RMSError * 255.0f);
    fname = prefix + "_deblurSpsReg_" + std::to_string(RMSError * 255.0f) +
            fileExtension;
    printf("Done, RMS Error: %f\n", RMSError * 255.0f);
    writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                     deblurImg[2]);
  }

  return EXIT_SUCCESS;
}
