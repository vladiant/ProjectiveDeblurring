#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "BilateralLaplacianRegularizer.hpp"
#include "BilateralRegularizer.hpp"
#include "BlurUtils.hpp"
#include "DeblurParameters.hpp"
#include "EmptyErrorCalculator.hpp"
#include "EmptyRegularizer.hpp"
#include "GaussianNoiseGenerator.hpp"
#include "ImResize.h"
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
  RLDeblurrer rLDeblurrer{blurGenerator, emptyErrorCalculator};

  {
    printf("Bilateral Regularization Algorithm:\n");

    BilateralRegularizer bilateralRegularizer;
    DeblurParameters bilateralRLParams{.Niter = 100, .bPoisson = true};
    RLDeblurrer rLDeblurrerBilateralReg{blurGenerator, emptyErrorCalculator};

    // rLDeblurrerBilateralReg.deblurRgb(
    //     bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth,
    //     blurheight, deblurImg[0].data(), deblurImg[1].data(),
    //     deblurImg[2].data(), width, height, DeblurParameters{.Niter = 100,
    //     .bPoisson = true}, bilateralRegularizer, 0.5f);
    rLDeblurrerBilateralReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralRLParams, bilateralRegularizer, 1.0f);
    rLDeblurrerBilateralReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralRLParams, bilateralRegularizer, 0.5f);
    rLDeblurrerBilateralReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralRLParams, bilateralRegularizer, 0.25f);
    rLDeblurrerBilateralReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralRLParams, bilateralRegularizer, 0.125f);

    DeblurParameters rLParams{.Niter = 100, .bPoisson = true};
    rLDeblurrer.deblurRgb(bImg[0].data(), bImg[1].data(), bImg[2].data(),
                          blurwidth, blurheight, deblurImg[0].data(),
                          deblurImg[1].data(), deblurImg[2].data(), width,
                          height, rLParams, emptyRegularizer, 0.0);
    RMSError = errorCalculator.calculateErrorRgb(
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height);
    //   sprintf(fname, "%s_deblurBilateralReg_%f.bmp", prefix, RMSError *
    //   255.0f);
    fname = prefix + "_deblurBilateralReg_" +
            std::to_string(RMSError * 255.0f) + fileExtension;
    printf("Done, RMS Error: %f\n", RMSError * 255.0f);
    writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                     deblurImg[2]);
  }

  {
    printf("Bilateral Laplacian Regularization Algorithm:\n");

    BilateralLaplacianRegularizer bilateralLaplacianRegularizer;
    DeblurParameters bilateralLaplacianRLParams{.Niter = 100, .bPoisson = true};
    RLDeblurrer rLDeblurrerBilateralLaplReg{blurGenerator,
                                            emptyErrorCalculator};

    // rLDeblurrerBilateralLaplReg.deblurRgb(
    //     bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth,
    //     blurheight, deblurImg[0].data(), deblurImg[1].data(),
    //     deblurImg[2].data(), width, height, DeblurParameters{.Niter = 500,
    //     .bPoisson = true}, bilateralLaplacianRegularizer, 0.5f);
    rLDeblurrerBilateralLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralLaplacianRLParams, bilateralLaplacianRegularizer,
        1.0f);
    rLDeblurrerBilateralLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralLaplacianRLParams, bilateralLaplacianRegularizer,
        0.5f);
    rLDeblurrerBilateralLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralLaplacianRLParams, bilateralLaplacianRegularizer,
        0.25f);
    rLDeblurrerBilateralLaplReg.deblurRgb(
        bImg[0].data(), bImg[1].data(), bImg[2].data(), blurwidth, blurheight,
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height, bilateralLaplacianRLParams, bilateralLaplacianRegularizer,
        0.125f);

    DeblurParameters rLParams{.Niter = 100, .bPoisson = true};
    rLDeblurrer.deblurRgb(bImg[0].data(), bImg[1].data(), bImg[2].data(),
                          blurwidth, blurheight, deblurImg[0].data(),
                          deblurImg[1].data(), deblurImg[2].data(), width,
                          height, rLParams, emptyRegularizer, 0.0);
    RMSError = errorCalculator.calculateErrorRgb(
        deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(), width,
        height);
    //   sprintf(fname, "%s_deblurBilateralLapReg_%f.bmp", prefix, RMSError *
    //   255.0f);
    fname = prefix + "_deblurBilateralLapReg_" +
            std::to_string(RMSError * 255.0f) + fileExtension;
    printf("Done, RMS Error: %f\n", RMSError * 255.0f);
    writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                     deblurImg[2]);
  }

  ///////////////////////////////////
  // Projective Motion RL Multi Scale Gray
  {
    printf("Multiscale Algorithm:\n");

    ProjectiveMotionRLMultiScaleGray rLDeblurrerMultiscale;

    for (int i = 0; i < MotionBlurImageGenerator::NumSamples; i++) {
      rLDeblurrerMultiscale.Hmatrix[i] = blurGenerator.Hmatrix[i];
      Homography::MatrixInverse(rLDeblurrerMultiscale.Hmatrix[i].Hmatrix,
                                rLDeblurrerMultiscale.IHmatrix[i].Hmatrix);
    }

    rLDeblurrerMultiscale.ProjectiveMotionRLDeblurMultiScaleGray(
        bImg[0].data(), blurwidth, blurheight, deblurImg[0].data(), width,
        height, 100, 5, true);

    fname = prefix + "_deblurMultiscale_" + std::to_string(RMSError * 255.0f) +
            fileExtension;
    printf("Done, RMS Error: %f\n", RMSError * 255.0f);
    writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[0],
                     deblurImg[0]);
  }

  ///////////////////////////////////
  if (false) {
    printf("Testing for Convergence\n");

    //   sprintf(fname, "ConvergencePoisson%s.txt", prefix);
    fname = "ConvergencePoisson" + prefix + ".txt";
    {
      std::fstream fp(fname, std::fstream::out);
      for (int iteration = 0; iteration < 5000; iteration++) {
        DeblurParameters rLParams{.Niter = 1, .bPoisson = true};
        rLDeblurrer.deblurRgb(bImg[0].data(), bImg[1].data(), bImg[2].data(),
                              blurwidth, blurheight, deblurImg[0].data(),
                              deblurImg[1].data(), deblurImg[2].data(), width,
                              height, rLParams, emptyRegularizer, 0.0);
        RMSError = errorCalculator.calculateErrorRgb(
            deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(),
            width, height);
        fp << std::setprecision(12) << RMSError * 255.0f << '\n';
        if (iteration % 100 == 0 || iteration == 20 || iteration == 50) {
          //   sprintf(fname.c_str(), "%s_deblurBasic_pitr%d_%f.bmp", prefix,
          //   iteration, RMSError * 255.0f);
          fname = prefix + "_deblurBasic_pitr" + std::to_string(iteration) +
                  "_" + std::to_string(RMSError * 255.0f) + fileExtension;
          writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                           deblurImg[2]);
        }
      }
    }

    //   sprintf(fname, "ConvergenceGaussian%s.txt", prefix);
    fname = "ConvergenceGaussian" + prefix + ".txt";
    {
      std::fstream fp(fname, std::fstream::out);
      for (int iteration = 0; iteration < 5000; iteration++) {
        DeblurParameters rLParams{.Niter = 1, .bPoisson = false};
        rLDeblurrer.deblurRgb(bImg[0].data(), bImg[1].data(), bImg[2].data(),
                              blurwidth, blurheight, deblurImg[0].data(),
                              deblurImg[1].data(), deblurImg[2].data(), width,
                              height, rLParams, emptyRegularizer, 0.0);
        RMSError = errorCalculator.calculateErrorRgb(
            deblurImg[0].data(), deblurImg[1].data(), deblurImg[2].data(),
            width, height);
        fp << std::setprecision(12) << RMSError * 255.0f << '\n';
        if (iteration % 100 == 0 || iteration == 20 || iteration == 50) {
          //   sprintf(fname, "%s_deblurBasic_gitr%d_%f.bmp", prefix, iteration,
          //   RMSError * 255.0f);
          fname = prefix + "_deblurBasic_gitr" + std::to_string(iteration) +
                  "_" + std::to_string(RMSError * 255.0f) + fileExtension;
          writeBMPchannels(fname, width, height, deblurImg[0], deblurImg[1],
                           deblurImg[2]);
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
