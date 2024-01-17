#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "ImResize.h"
#include "ProjectiveMotionRL.h"
#include "bitmap.h"

int main(int /*argc*/, char* /*argv*/[]) {
  int width = 0, height = 0;
  std::vector<float> fImg;
  std::string prefix = "doll";
  // memcpy(prefix, argv[1], strlen(argv[1])+1);
  std::string fname;
  fname = prefix + ".bmp";
  printf("Load Image: %s\n", fname.c_str());
  readBMP(fname, fImg, width, height);
  int blurwidth = width, blurheight = height;

  std::vector<float> bImg;
  std::vector<float> deblurImg;
  std::vector<float> intermediatedeblurImg;
  std::vector<float> inputWeight;
  std::vector<float> outputWeight(width * height);
  float RMSError = NAN;
  bImg.resize(width * height * 3);
  deblurImg.resize(width * height * 3);
  intermediatedeblurImg.resize(width * height * 3);

  ///////////////////////////////////
  printf("Set Projective Model Parameter\n");
  ProjectiveMotionRL m_ProjectiveMotionRL;

  m_ProjectiveMotionRL.SetGlobalParameters(
      10, 1.2f, 0.0003f, 0.0006f, 10,
      20);  // This is parameter setting for doll example

  // m_ProjectiveMotionRL.SetGlobalParameters(-10, 1.1f, 0.0004f,0.0002f, 20,
  // -15); //This is parameter setting for Cameraman convergence example

  // Testing case for rotational motion
  // m_ProjectiveMotionRL.SetGlobalRotation(10);

  // Testing case for zooming motion
  // m_ProjectiveMotionRL.SetGlobalScaling(1.2f);

  // Testing case for translational motion
  if (false) {
    float deltadx = 0.8f;
    for (int i = 0; i < ProjectiveMotionRL::NumSamples; i++) {
      float dy =
          5.0f * sin((float)(i) / ProjectiveMotionRL::NumSamples * 2 * M_PI);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][0] = 1;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][1] = 0;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][2] = i * deltadx;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][0] = 0;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][1] = 1;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][2] = dy;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][0] = 0;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][1] = 0;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][2] = 1;
      Homography::MatrixInverse(m_ProjectiveMotionRL.Hmatrix[i].Hmatrix,
                                m_ProjectiveMotionRL.IHmatrix[i].Hmatrix);
    }
  }

  // Testing case for projective motion
  // m_ProjectiveMotionRL.SetGlobalPerspective(0.001f,0.001f);

  // Adjust the parameters for generating test case 1-15
  if (false) {
    float deltadx = 0.8f;
    float deltascaling = 0.2f / ProjectiveMotionRL::NumSamples;
    float deltapx = 0.001f / ProjectiveMotionRL::NumSamples;
    float deltapy = 0.001f / ProjectiveMotionRL::NumSamples;
    float deltadegree =
        (10.0f * M_PI / 180.0f) / ProjectiveMotionRL::NumSamples;
    for (int i = 0; i < ProjectiveMotionRL::NumSamples; i++) {
      float dy =
          5.0f * sin((float)(i) / ProjectiveMotionRL::NumSamples * 2 * M_PI);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][0] =
          (1 + i * deltascaling) * cos(deltadegree * i);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[0][2] = i * deltadx;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][1] =
          (1 + i * deltascaling) * cos(deltadegree * i);
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[1][2] = dy;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][0] = i * deltapx;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][1] = i * deltapy;
      m_ProjectiveMotionRL.Hmatrix[i].Hmatrix[2][2] = 1;
      Homography::MatrixInverse(m_ProjectiveMotionRL.Hmatrix[i].Hmatrix,
                                m_ProjectiveMotionRL.IHmatrix[i].Hmatrix);
    }
  }

  ///////////////////////////////////
  m_ProjectiveMotionRL.SetGroundTruthImgRgb(
      fImg.data(), width,
      height);  // This is for error computation

  ///////////////////////////////////
  printf("Generate Motion Blurred Image\n");
  m_ProjectiveMotionRL.GenerateMotionBlurImgRgb(
      fImg.data(), inputWeight.data(), width, height, bImg.data(),
      outputWeight.data(), blurwidth, blurheight, true);

  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(fImg.data(), bImg.data(),
                                                     width, height);
  //   sprintf(fname, "%s_blur_%.6f.bmp", prefix, RMSError * 255.0f);
  fname = prefix + "_blur_" + std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Save Blurred Image to: %s\n", fname.c_str());
  writeBMP(fname, blurwidth, blurheight, bImg);

  // Add noise
  float sigma = 2.0f;
  m_ProjectiveMotionRL.gaussianNoiseRGB(bImg.data(), width, height, sigma);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(fImg.data(), bImg.data(),
                                                     width, height);
  //   sprintf(fname, "%s_blur_noise_sigma%.2f_%.6f.bmp", prefix, sigma,
  //   RMSError * 255.0f);
  fname = prefix + "_blur_noise_sigma" + std::to_string(sigma) + "_" +
          std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Save Blurred Image to: %s\n", fname.c_str());
  writeBMP(fname, blurwidth, blurheight, bImg);

  ///////////////////////////////////
  // Main Deblurring algorithm
  printf("Initial Estimation is the blur image\n");
  ImChoppingRGB(bImg.data(), blurwidth, blurheight, deblurImg.data(), width,
                height);
  //	memset(deblurImg[0].data(), 0, width*height*sizeof(float));
  //	memset(deblurImg[1].data(), 0, width*height*sizeof(float));
  //	memset(deblurImg[2].data(), 0, width*height*sizeof(float));

  // Levin et. al. Siggraph07's matlab implementation also take around 400
  // iterations Sadly, the algorithm needs such a lot of iterations to produce
  // good results Most running time were spent on bicubic interpolation, it
  // would be much faster if this step was implemented in GPU...

  // Load Initial Guess, if you have...
  //   readBMP("", deblurImg[0], deblurImg[1], deblurImg[2], width, height);
  memcpy(intermediatedeblurImg.data(), deblurImg.data(),
         width * height * sizeof(float) * 3);

  printf("Basic Algorithm:\n");
  memcpy(deblurImg.data(), intermediatedeblurImg.data(),
         width * height * sizeof(float) * 3);

  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(bImg.data(), blurwidth,
                                                   blurheight, deblurImg.data(),
                                                   width, height, 500, true);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
      fImg.data(), deblurImg.data(), width, height);
  //   sprintf(fname, "%s_deblurBasic_%f.bmp", prefix, RMSError * 255.0f);
  fname = prefix + "_deblurBasic_" + std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);
  writeBMP(fname, width, height, deblurImg);

  printf("TV Regularization Algorithm:\n");
  memcpy(deblurImg.data(), intermediatedeblurImg.data(),
         width * height * sizeof(float) * 3);

  // Gradually decrease the regularization weight, otherwise, the result will be
  // over smooth. Actually, the following regularization produce similar
  // results....
  //   m_ProjectiveMotionRL.ProjectiveMotionRLDeblurTVReg(
  //       bImg.data(), blurwidth, blurheight,
  //       deblurImg.data(), width, height, 500, true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurTVRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 1.0f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurTVRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurTVRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.25f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurTVRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.125f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(bImg.data(), blurwidth,
                                                   blurheight, deblurImg.data(),
                                                   width, height, 100, true);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
      fImg.data(), deblurImg.data(), width, height);
  //   sprintf(fname, "%s_deblurTVReg_%f.bmp", prefix, RMSError * 255.0f);
  fname = prefix + "_deblurTVReg_" + std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);
  writeBMP(fname, width, height, deblurImg);

  printf("Laplacian Regularization Algorithm:\n");
  memcpy(deblurImg.data(), intermediatedeblurImg.data(),
         width * height * sizeof(float) * 3);

  //   m_ProjectiveMotionRL.ProjectiveMotionRLDeblurSpsReg(
  //       bImg.data(), blurwidth, blurheight,
  //   deblurImg.data(), width, height, 500, true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurSpsRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 1.0f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurSpsRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurSpsRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.25f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurSpsRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.125f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(bImg.data(), blurwidth,
                                                   blurheight, deblurImg.data(),
                                                   width, height, 100, true);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
      fImg.data(), deblurImg.data(), width, height);
  //   sprintf(fname, "%s_deblurSpsReg_%f.bmp", prefix, RMSError * 255.0f);
  fname =
      prefix + "_deblurSpsReg_" + std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);
  writeBMP(fname, width, height, deblurImg);

  printf("Bilateral Regularization Algorithm:\n");
  memcpy(deblurImg.data(), intermediatedeblurImg.data(),
         width * height * sizeof(float) * 3);

  //   m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralReg(
  //       bImg.data(), blurwidth, blurheight,
  //   deblurImg.data(), width, height, 500, true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 1.0f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.25f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.125f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(bImg.data(), blurwidth,
                                                   blurheight, deblurImg.data(),
                                                   width, height, 100, true);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
      fImg.data(), deblurImg.data(), width, height);
  //   sprintf(fname, "%s_deblurBilateralReg_%f.bmp", prefix, RMSError *
  //   255.0f);
  fname = prefix + "_deblurBilateralReg_" + std::to_string(RMSError * 255.0f) +
          ".bmp";
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);
  writeBMP(fname, width, height, deblurImg);

  printf("Bilateral Laplacian Regularization Algorithm:\n");
  memcpy(deblurImg.data(), intermediatedeblurImg.data(),
         width * height * sizeof(float) * 3);

  //   m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralLapReg(
  //       bImg.data(), blurwidth, blurheight,
  //   deblurImg.data(), width, height, 500, true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralLapRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 1.0f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralLapRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.5f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralLapRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.25f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurBilateralLapRegRgb(
      bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height, 100,
      true, 0.125f);
  m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(bImg.data(), blurwidth,
                                                   blurheight, deblurImg.data(),
                                                   width, height, 100, true);
  RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
      fImg.data(), deblurImg.data(), width, height);
  //   sprintf(fname, "%s_deblurBilateralLapReg_%f.bmp", prefix, RMSError *
  //   255.0f);
  fname = prefix + "_deblurBilateralLapReg_" +
          std::to_string(RMSError * 255.0f) + ".bmp";
  printf("Done, RMS Error: %f\n", RMSError * 255.0f);
  writeBMP(fname, width, height, deblurImg);

  ///////////////////////////////////
  if (false) {
    printf("Testing for Convergence\n");
    memcpy(deblurImg.data(), intermediatedeblurImg.data(),
           width * height * sizeof(float) * 3);

    //   sprintf(fname, "ConvergencePoisson%s.txt", prefix);
    fname = "ConvergencePoisson" + prefix + ".txt";
    {
      std::fstream fp(fname, std::fstream::out);
      for (int iteration = 0; iteration < 5000; iteration++) {
        m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(
            bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height,
            1, true);
        RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
            fImg.data(), deblurImg.data(), width, height);
        fp << std::setprecision(12) << RMSError * 255.0f << '\n';
        if (iteration % 100 == 0 || iteration == 20 || iteration == 50) {
          //   sprintf(fname.c_str(), "%s_deblurBasic_pitr%d_%f.bmp", prefix,
          //   iteration, RMSError * 255.0f);
          fname = prefix + "_deblurBasic_pitr" + std::to_string(iteration) +
                  "_" + std::to_string(RMSError * 255.0f) + ".bmp";
          writeBMP(fname, width, height, deblurImg);
        }
      }
    }

    memcpy(deblurImg.data(), intermediatedeblurImg.data(),
           width * height * sizeof(float) * 3);
    //   sprintf(fname, "ConvergenceGaussian%s.txt", prefix);
    fname = "ConvergenceGaussian" + prefix + ".txt";
    {
      std::fstream fp(fname, std::fstream::out);
      for (int iteration = 0; iteration < 5000; iteration++) {
        m_ProjectiveMotionRL.ProjectiveMotionRLDeblurRgb(
            bImg.data(), blurwidth, blurheight, deblurImg.data(), width, height,
            1, false);
        RMSError = m_ProjectiveMotionRL.ComputeRMSErrorRGB(
            fImg.data(), deblurImg.data(), width, height);
        fp << std::setprecision(12) << RMSError * 255.0f << '\n';
        if (iteration % 100 == 0 || iteration == 20 || iteration == 50) {
          //   sprintf(fname, "%s_deblurBasic_gitr%d_%f.bmp", prefix, iteration,
          //   RMSError * 255.0f);
          fname = prefix + "_deblurBasic_gitr" + std::to_string(iteration) +
                  "_" + std::to_string(RMSError * 255.0f) + ".bmp";
          writeBMP(fname, width, height, deblurImg);
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
