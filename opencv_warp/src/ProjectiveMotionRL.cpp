#include "ProjectiveMotionRL.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/imgproc.hpp>

#include "BicubicInterpolation.h"
#include "ImResize.h"
#include "bitmap.h"
#include "warping.h"

float ProjectiveMotionRL::getSpsWeight(float aValue) const {
  if (!std::isfinite(aValue)) {
    return 0.0f;
  }

  return SpsTable[static_cast<int>(std::abs(aValue) * 255.0f)];
}

void ProjectiveMotionRL::WarpImage(float* InputImg, float* inputWeight,
                                   int iwidth, int iheight, float* OutputImg,
                                   float* outputWeight, int width, int height,
                                   int i) {
  if (i >= 0 && i < NumSamples) {
    warpImage(InputImg, inputWeight, iwidth, iheight, OutputImg, outputWeight,
              width, height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImage(InputImg, inputWeight, iwidth, iheight, OutputImg, outputWeight,
              width, height, Hmatrix[-i]);
  }
}

void ProjectiveMotionRL::WarpImage(float* InputImg, int iwidth, int iheight,
                                   float* OutputImg, float* outputWeight,
                                   int width, int height, int i) {
  int x = 0, y = 0, index = 0;
  float fx = NAN, fy = NAN;
  if (i >= 0 && i < NumSamples) {
    float woffset = width * 0.5f, hoffset = height * 0.5f;
    float iwoffset = iwidth * 0.5f, ihoffset = iheight * 0.5f;

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        fx = x - woffset;
        fy = y - hoffset;
        IHmatrix[i].Transform(fx, fy);  // Inverse mapping, use inverse instead
        fx += iwoffset;
        fy += ihoffset;

        if (fx >= 0 && fx < iwidth - 1 && fy >= 0 && fy < iheight - 1) {
          outputWeight[index] = 1.01f;
        } else {
          outputWeight[index] = 0.01f;
        }

        if (fx < 0) fx = 0;
        if (fy < 0) fy = 0;
        if (fx >= iwidth - 1.001f) fx = iwidth - 1.001f;
        if (fy >= iheight - 1.001f) fy = iheight - 1.001f;

        // OutputImg[index] =
        //     ReturnInterpolatedValueFast(fx, fy, InputImg, iwidth, iheight);
      }
    }

    const cv::Mat inImg(height, width, CV_32FC1, InputImg);
    const cv::Mat outImg(height, width, CV_32FC1, OutputImg);
    const cv::Mat hMat(3, 3, CV_32FC1, IHmatrix[i].Hmatrix);

    float transf[9] = {1.0, 0.0, -woffset, 0.0, 1.0, -hoffset, 0.0, 0.0, 1.0};
    const cv::Mat coordTransf(3, 3, CV_32FC1, transf);
    float backTransf[9] = {1.0, 0.0, woffset, 0.0, 1.0, hoffset, 0.0, 0.0, 1.0};
    const cv::Mat coordBackTransf(3, 3, CV_32FC1, backTransf);
    const cv::Mat perspMat = coordBackTransf * hMat * coordTransf;

    cv::warpPerspective(inImg, outImg, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);

  } else if (i < 0 && i > -NumSamples) {
    float woffset = width * 0.5f, hoffset = height * 0.5f;
    float iwoffset = iwidth * 0.5f, ihoffset = iheight * 0.5f;

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        fx = x - woffset;
        fy = y - hoffset;
        Hmatrix[-i].Transform(fx, fy);  // Inverse mapping, use inverse instead
        fx += iwoffset;
        fy += ihoffset;

        if (fx >= 0 && fx < iwidth - 1 && fy >= 0 && fy < iheight - 1) {
          outputWeight[index] = 1.01f;
        } else {
          outputWeight[index] = 0.01f;
        }

        if (fx < 0) fx = 0;
        if (fy < 0) fy = 0;
        if (fx >= iwidth - 1.001f) fx = iwidth - 1.001f;
        if (fy >= iheight - 1.001f) fy = iheight - 1.001f;

        // OutputImg[index] =
        //     ReturnInterpolatedValueFast(fx, fy, InputImg, iwidth, iheight);
      }
    }

    const cv::Mat inImg(height, width, CV_32FC1, InputImg);
    const cv::Mat outImg(height, width, CV_32FC1, OutputImg);
    const cv::Mat hMat(3, 3, CV_32FC1, Hmatrix[-i].Hmatrix);

    float transf[9] = {1.0, 0.0, -woffset, 0.0, 1.0, -hoffset, 0.0, 0.0, 1.0};
    const cv::Mat coordTransf(3, 3, CV_32FC1, transf);
    float backTransf[9] = {1.0, 0.0, woffset, 0.0, 1.0, hoffset, 0.0, 0.0, 1.0};
    const cv::Mat coordBackTransf(3, 3, CV_32FC1, backTransf);
    const cv::Mat perspMat = coordBackTransf * hMat * coordTransf;

    cv::warpPerspective(inImg, outImg, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  }
}

void ProjectiveMotionRL::WarpImage(float* InputImgR, float* InputImgG,
                                   float* InputImgB, float* inputWeight,
                                   int iwidth, int iheight, float* OutputImgR,
                                   float* OutputImgG, float* OutputImgB,
                                   float* outputWeight, int width, int height,
                                   int i) {
  if (i >= 0 && i < NumSamples) {
    warpImage(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
              OutputImgR, OutputImgG, OutputImgB, outputWeight, width, height,
              IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImage(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
              OutputImgR, OutputImgG, OutputImgB, outputWeight, width, height,
              Hmatrix[-i]);
  }
}

void ProjectiveMotionRL::GenerateMotionBlurImg(float* InputImg, int iwidth,
                                               int iheight, float* BlurImg,
                                               float* outputWeight, int width,
                                               int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  if (WarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImg, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImage(InputImg, iwidth, iheight, WarpImgBuffer.data(),
                WarpWeightBuffer.data(), width, height, i);
    } else {
      WarpImage(InputImg, iwidth, iheight, WarpImgBuffer.data(),
                WarpWeightBuffer.data(), width, height, -i);
    }
    for (index = 0; index < totalpixel; index++) {
      BlurImg[index] += WarpImgBuffer[index] * WarpWeightBuffer[index];
      outputWeight[index] += WarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImg[index] /= outputWeight[index];
  }
}

void ProjectiveMotionRL::GenerateMotionBlurImg(float* InputImg,
                                               float* inputWeight, int iwidth,
                                               int iheight, float* BlurImg,
                                               float* outputWeight, int width,
                                               int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  if (WarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImg, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImage(InputImg, inputWeight, iwidth, iheight, WarpImgBuffer.data(),
                WarpWeightBuffer.data(), width, height, i);
    } else {
      WarpImage(InputImg, inputWeight, iwidth, iheight, WarpImgBuffer.data(),
                WarpWeightBuffer.data(), width, height, -i);
    }
    for (index = 0; index < totalpixel; index++) {
      BlurImg[index] += WarpImgBuffer[index] * WarpWeightBuffer[index];
      outputWeight[index] += WarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImg[index] /= outputWeight[index];
  }
}

void ProjectiveMotionRL::GenerateMotionBlurImg(
    float* InputImgR, float* InputImgG, float* InputImgB, float* inputWeight,
    int iwidth, int iheight, float* BlurImgR, float* BlurImgG, float* BlurImgB,
    float* outputWeight, int width, int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  if (WarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImgR, 0, totalpixel * sizeof(float));
  memset(BlurImgG, 0, totalpixel * sizeof(float));
  memset(BlurImgB, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImage(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
                WarpImgBufferR.data(), WarpImgBufferG.data(),
                WarpImgBufferB.data(), WarpWeightBuffer.data(), width, height,
                i);
    } else {
      WarpImage(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
                WarpImgBufferR.data(), WarpImgBufferG.data(),
                WarpImgBufferB.data(), WarpWeightBuffer.data(), width, height,
                -i);
    }

    for (index = 0; index < totalpixel; index++) {
      BlurImgR[index] += WarpImgBufferR[index] * WarpWeightBuffer[index];
      BlurImgG[index] += WarpImgBufferG[index] * WarpWeightBuffer[index];
      BlurImgB[index] += WarpImgBufferB[index] * WarpWeightBuffer[index];
      outputWeight[index] += WarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImgR[index] /= outputWeight[index];
    BlurImgG[index] /= outputWeight[index];
    BlurImgB[index] /= outputWeight[index];
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblur(float* BlurImg, int iwidth,
                                                  int iheight, float* DeblurImg,
                                                  int width, int height,
                                                  int Niter, bool bPoisson) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImg, InputWeight, width, height,
                          BlurImgBuffer.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBuffer[index] > 0.001f) {
            DeltaImg[index] = BlurImg[index] / BlurImgBuffer[index];
          } else {
            DeltaImg[index] = BlurImg[index] / 0.001f;
          }
        } else {
          DeltaImg[index] = BlurImg[index] - BlurImgBuffer[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, ErrorImgBuffer.data(),
                          ErrorWeightBuffer.data(), width, height, false);
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImg[index] *= ErrorImgBuffer[index];
        } else {
          DeblurImg[index] += ErrorImgBuffer[index];
        }

        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error: %f\n",
           ComputeRMSError(GroundTruthImg, DeblurImg, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblur(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgR(iwidth * iheight);
  std::vector<float> DeltaImgG(iwidth * iheight);
  std::vector<float> DeltaImgB(iwidth * iheight);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                          width, height, BlurImgBufferR.data(),
                          BlurImgBufferG.data(), BlurImgBufferB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferR[index] > 0.001f) {
            DeltaImgR[index] = BlurImgR[index] / BlurImgBufferR[index];
          } else {
            DeltaImgR[index] = BlurImgR[index] / 0.001f;
          }
          if (BlurImgBufferG[index] > 0.001f) {
            DeltaImgG[index] = BlurImgG[index] / BlurImgBufferG[index];
          } else {
            DeltaImgG[index] = BlurImgG[index] / 0.001f;
          }
          if (BlurImgBufferB[index] > 0.001f) {
            DeltaImgB[index] = BlurImgB[index] / BlurImgBufferB[index];
          } else {
            DeltaImgB[index] = BlurImgB[index] / 0.001f;
          }
        } else {
          DeltaImgR[index] = BlurImgR[index] - BlurImgBufferR[index];
          DeltaImgG[index] = BlurImgG[index] - BlurImgBufferG[index];
          DeltaImgB[index] = BlurImgB[index] - BlurImgBufferB[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImgR.data(), DeltaImgG.data(), DeltaImgB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight,
                          ErrorImgBufferR.data(), ErrorImgBufferG.data(),
                          ErrorImgBufferB.data(), ErrorWeightBuffer.data(),
                          width, height, false);
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgR[index] *= ErrorImgBufferR[index];
          DeblurImgG[index] *= ErrorImgBufferG[index];
          DeblurImgB[index] *= ErrorImgBufferB[index];
        } else {
          DeblurImgR[index] += ErrorImgBufferR[index];
          DeblurImgG[index] += ErrorImgBufferG[index];
          DeblurImgB[index] += ErrorImgBufferB[index];
        }

        DeblurImgR[index] = std::clamp(DeblurImgR[index], 0.0f, 1.0f);
        DeblurImgG[index] = std::clamp(DeblurImgG[index], 0.0f, 1.0f);
        DeblurImgB[index] = std::clamp(DeblurImgB[index], 0.0f, 1.0f);
      }
    }
#ifdef __SHOWERROR__
    printf("RMS Error R: %f\n",
           ComputeRMSError(GroundTruthImgR, DeblurImgR, width, height));
    printf("RMS Error G: %f\n",
           ComputeRMSError(GroundTruthImgG, DeblurImgG, width, height));
    printf("RMS Error B: %f\n",
           ComputeRMSError(GroundTruthImgB, DeblurImgB, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurTVReg(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);
  std::vector<float> DxImg(width * height);
  std::vector<float> DyImg(width * height);
  std::vector<float> DxxImg(width * height);
  std::vector<float> DyyImg(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImg, InputWeight, width, height,
                          BlurImgBuffer.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBuffer[index] > 0.001f) {
            DeltaImg[index] = BlurImg[index] / BlurImgBuffer[index];
          } else {
            DeltaImg[index] = BlurImg[index] / 0.001f;
          }
        } else {
          DeltaImg[index] = BlurImg[index] - BlurImgBuffer[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, ErrorImgBuffer.data(),
                          ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImage(DeblurImg, width, height, DxImg.data(), DyImg.data(),
                         true);
    for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
      for (x = 0; x < width; x++, index++) {
        if (DxImg[index] > 0) DxImg[index] = 1.0f / 255.0f;
        if (DxImg[index] < 0) DxImg[index] = -1.0f / 255.0f;
        if (DyImg[index] > 0) DyImg[index] = 1.0f / 255.0f;
        if (DyImg[index] < 0) DyImg[index] = -1.0f / 255.0f;
      }
    }
    ComputeGradientXImage(DxImg.data(), width, height, DxxImg.data(), false);
    ComputeGradientYImage(DyImg.data(), width, height, DyyImg.data(), false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImg[index] *= ErrorImgBuffer[index] /
                              (1 + lambda * (DxxImg[index] + DyyImg[index]));
        } else {
          DeblurImg[index] +=
              ErrorImgBuffer[index] - lambda * (DxxImg[index] + DyyImg[index]);
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error: %f\n",
           ComputeRMSError(GroundTruthImg, DeblurImg, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurTVReg(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgR(iwidth * iheight);
  std::vector<float> DeltaImgG(iwidth * iheight);
  std::vector<float> DeltaImgB(iwidth * iheight);

  std::vector<float> DxImgR(width * height);
  std::vector<float> DyImgR(width * height);
  std::vector<float> DxxImgR(width * height);
  std::vector<float> DyyImgR(width * height);
  std::vector<float> DxImgG(width * height);
  std::vector<float> DyImgG(width * height);
  std::vector<float> DxxImgG(width * height);
  std::vector<float> DyyImgG(width * height);
  std::vector<float> DxImgB(width * height);
  std::vector<float> DyImgB(width * height);
  std::vector<float> DxxImgB(width * height);
  std::vector<float> DyyImgB(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                          width, height, BlurImgBufferR.data(),
                          BlurImgBufferG.data(), BlurImgBufferB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferR[index] > 0.001f) {
            DeltaImgR[index] = BlurImgR[index] / BlurImgBufferR[index];
          } else {
            DeltaImgR[index] = BlurImgR[index] / 0.001f;
          }
          if (BlurImgBufferG[index] > 0.001f) {
            DeltaImgG[index] = BlurImgG[index] / BlurImgBufferG[index];
          } else {
            DeltaImgG[index] = BlurImgG[index] / 0.001f;
          }
          if (BlurImgBufferB[index] > 0.001f) {
            DeltaImgB[index] = BlurImgB[index] / BlurImgBufferB[index];
          } else {
            DeltaImgB[index] = BlurImgB[index] / 0.001f;
          }
        } else {
          DeltaImgR[index] = BlurImgR[index] - BlurImgBufferR[index];
          DeltaImgG[index] = BlurImgG[index] - BlurImgBufferG[index];
          DeltaImgB[index] = BlurImgB[index] - BlurImgBufferB[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImgR.data(), DeltaImgG.data(), DeltaImgB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight,
                          ErrorImgBufferR.data(), ErrorImgBufferG.data(),
                          ErrorImgBufferB.data(), ErrorWeightBuffer.data(),
                          width, height, false);

    ComputeGradientImage(DeblurImgR, width, height, DxImgR.data(),
                         DyImgR.data(), true);
    ComputeGradientImage(DeblurImgG, width, height, DxImgG.data(),
                         DyImgG.data(), true);
    ComputeGradientImage(DeblurImgB, width, height, DxImgB.data(),
                         DyImgB.data(), true);
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (DxImgR[index] > 0) DxImgR[index] = 1.0f / 255.0f;
        if (DxImgR[index] < 0) DxImgR[index] = -1.0f / 255.0f;
        if (DyImgR[index] > 0) DyImgR[index] = 1.0f / 255.0f;
        if (DyImgR[index] < 0) DyImgR[index] = -1.0f / 255.0f;
        if (DxImgG[index] > 0) DxImgG[index] = 1.0f / 255.0f;
        if (DxImgG[index] < 0) DxImgG[index] = -1.0f / 255.0f;
        if (DyImgG[index] > 0) DyImgG[index] = 1.0f / 255.0f;
        if (DyImgG[index] < 0) DyImgG[index] = -1.0f / 255.0f;
        if (DxImgB[index] > 0) DxImgB[index] = 1.0f / 255.0f;
        if (DxImgB[index] < 0) DxImgB[index] = -1.0f / 255.0f;
        if (DyImgB[index] > 0) DyImgB[index] = 1.0f / 255.0f;
        if (DyImgB[index] < 0) DyImgB[index] = -1.0f / 255.0f;
      }
    }
    ComputeGradientXImage(DxImgR.data(), width, height, DxxImgR.data(), false);
    ComputeGradientYImage(DyImgR.data(), width, height, DyyImgR.data(), false);
    ComputeGradientXImage(DxImgG.data(), width, height, DxxImgG.data(), false);
    ComputeGradientYImage(DyImgG.data(), width, height, DyyImgG.data(), false);
    ComputeGradientXImage(DxImgB.data(), width, height, DxxImgB.data(), false);
    ComputeGradientYImage(DyImgB.data(), width, height, DyyImgB.data(), false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgR[index] *= ErrorImgBufferR[index] /
                               (1 + lambda * (DxxImgR[index] + DyyImgR[index]));
          DeblurImgG[index] *= ErrorImgBufferG[index] /
                               (1 + lambda * (DxxImgG[index] + DyyImgG[index]));
          DeblurImgB[index] *= ErrorImgBufferB[index] /
                               (1 + lambda * (DxxImgB[index] + DyyImgB[index]));
        } else {
          DeblurImgR[index] += ErrorImgBufferR[index] -
                               lambda * (DxxImgR[index] + DyyImgR[index]);
          DeblurImgG[index] += ErrorImgBufferG[index] -
                               lambda * (DxxImgG[index] + DyyImgG[index]);
          DeblurImgB[index] += ErrorImgBufferB[index] -
                               lambda * (DxxImgB[index] + DyyImgB[index]);
        }
        DeblurImgR[index] = std::clamp(DeblurImgR[index], 0.0f, 1.0f);
        DeblurImgG[index] = std::clamp(DeblurImgG[index], 0.0f, 1.0f);
        DeblurImgB[index] = std::clamp(DeblurImgB[index], 0.0f, 1.0f);
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error R: %f\n",
           ComputeRMSError(GroundTruthImgR, DeblurImgR, width, height));
    printf("RMS Error G: %f\n",
           ComputeRMSError(GroundTruthImgG, DeblurImgG, width, height));
    printf("RMS Error B: %f\n",
           ComputeRMSError(GroundTruthImgB, DeblurImgB, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurSpsReg(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;
  float Wx = NAN, Wy = NAN;

  std::vector<float> DeltaImg(iwidth * iheight);

  std::vector<float> DxImg(width * height);
  std::vector<float> DyImg(width * height);
  std::vector<float> DxxImg(width * height);
  std::vector<float> DyyImg(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImg, InputWeight, width, height,
                          BlurImgBuffer.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBuffer[index] > 0.001f) {
            DeltaImg[index] = BlurImg[index] / BlurImgBuffer[index];
          } else {
            DeltaImg[index] = BlurImg[index] / 0.001f;
          }
        } else {
          DeltaImg[index] = BlurImg[index] - BlurImgBuffer[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, ErrorImgBuffer.data(),
                          ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImage(DeblurImg, width, height, DxImg.data(), DyImg.data(),
                         true);

    ComputeGradientXImage(DxImg.data(), width, height, DxxImg.data(), false);
    ComputeGradientYImage(DyImg.data(), width, height, DyyImg.data(), false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        Wx = getSpsWeight(DxImg[index]);
        Wy = getSpsWeight(DyImg[index]);

        if (bPoisson) {
          DeblurImg[index] =
              DeblurImg[index] /
              (1 - lambda * (Wx * DxxImg[index] + Wy * DyyImg[index])) *
              ErrorImgBuffer[index];
        } else {
          DeblurImg[index] = DeblurImg[index] + ErrorImgBuffer[index] -
                             lambda * (Wx * DxxImg[index] + Wy * DyyImg[index]);
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
        if (std::isnan(DeblurImg[index])) DeblurImg[index] = 0;
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error: %f\n",
           ComputeRMSError(GroundTruthImg, DeblurImg, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurSpsReg(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;
  float WxR = NAN, WyR = NAN, WxG = NAN, WyG = NAN, WxB = NAN, WyB = NAN;

  std::vector<float> DeltaImgR(iwidth * iheight);
  std::vector<float> DeltaImgG(iwidth * iheight);
  std::vector<float> DeltaImgB(iwidth * iheight);

  std::vector<float> DxImgR(width * height);
  std::vector<float> DyImgR(width * height);
  std::vector<float> DxxImgR(width * height);
  std::vector<float> DyyImgR(width * height);
  std::vector<float> DxImgG(width * height);
  std::vector<float> DyImgG(width * height);
  std::vector<float> DxxImgG(width * height);
  std::vector<float> DyyImgG(width * height);
  std::vector<float> DxImgB(width * height);
  std::vector<float> DyImgB(width * height);
  std::vector<float> DxxImgB(width * height);
  std::vector<float> DyyImgB(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                          width, height, BlurImgBufferR.data(),
                          BlurImgBufferG.data(), BlurImgBufferB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight, true);

    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferR[index] > 0.001f) {
            DeltaImgR[index] = BlurImgR[index] / BlurImgBufferR[index];
          } else {
            DeltaImgR[index] = BlurImgR[index] / 0.001f;
          }
          if (BlurImgBufferG[index] > 0.001f) {
            DeltaImgG[index] = BlurImgG[index] / BlurImgBufferG[index];
          } else {
            DeltaImgG[index] = BlurImgG[index] / 0.001f;
          }
          if (BlurImgBufferB[index] > 0.001f) {
            DeltaImgB[index] = BlurImgB[index] / BlurImgBufferB[index];
          } else {
            DeltaImgB[index] = BlurImgB[index] / 0.001f;
          }
        } else {
          DeltaImgR[index] = BlurImgR[index] - BlurImgBufferR[index];
          DeltaImgG[index] = BlurImgG[index] - BlurImgBufferG[index];
          DeltaImgB[index] = BlurImgB[index] - BlurImgBufferB[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImgR.data(), DeltaImgG.data(), DeltaImgB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight,
                          ErrorImgBufferR.data(), ErrorImgBufferG.data(),
                          ErrorImgBufferB.data(), ErrorWeightBuffer.data(),
                          width, height, false);

    ComputeGradientImage(DeblurImgR, width, height, DxImgR.data(),
                         DyImgR.data(), true);
    ComputeGradientImage(DeblurImgG, width, height, DxImgG.data(),
                         DyImgG.data(), true);
    ComputeGradientImage(DeblurImgB, width, height, DxImgB.data(),
                         DyImgB.data(), true);

    ComputeGradientXImage(DxImgR.data(), width, height, DxxImgR.data(), false);
    ComputeGradientYImage(DyImgR.data(), width, height, DyyImgR.data(), false);
    ComputeGradientXImage(DxImgG.data(), width, height, DxxImgG.data(), false);
    ComputeGradientYImage(DyImgG.data(), width, height, DyyImgG.data(), false);
    ComputeGradientXImage(DxImgB.data(), width, height, DxxImgB.data(), false);
    ComputeGradientYImage(DyImgB.data(), width, height, DyyImgB.data(), false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        WxR = getSpsWeight(DxImgR[index]);
        WyR = getSpsWeight(DyImgR[index]);
        WxG = getSpsWeight(DxImgG[index]);
        WyG = getSpsWeight(DyImgG[index]);
        WxB = getSpsWeight(DxImgB[index]);
        WyB = getSpsWeight(DyImgB[index]);

        if (bPoisson) {
          DeblurImgR[index] *=
              ErrorImgBufferR[index] /
              (1 + lambda * (WxR * DxxImgR[index] + WyR * DyyImgR[index]));
          DeblurImgG[index] *=
              ErrorImgBufferG[index] /
              (1 + lambda * (WxG * DxxImgG[index] + WyG * DyyImgG[index]));
          DeblurImgB[index] *=
              ErrorImgBufferB[index] /
              (1 + lambda * (WxB * DxxImgB[index] + WyB * DyyImgB[index]));
        } else {
          DeblurImgR[index] +=
              ErrorImgBufferR[index] -
              lambda * (WxR * DxxImgR[index] + WyR * DyyImgR[index]);
          DeblurImgG[index] +=
              ErrorImgBufferG[index] -
              lambda * (WxG * DxxImgG[index] + WyG * DyyImgG[index]);
          DeblurImgB[index] +=
              ErrorImgBufferB[index] -
              lambda * (WxB * DxxImgB[index] + WyB * DyyImgB[index]);
        }
        DeblurImgR[index] = std::clamp(DeblurImgR[index], 0.0f, 1.0f);
        DeblurImgG[index] = std::clamp(DeblurImgG[index], 0.0f, 1.0f);
        DeblurImgB[index] = std::clamp(DeblurImgB[index], 0.0f, 1.0f);

        if (std::isnan(DeblurImgR[index])) DeblurImgR[index] = 0;
        if (std::isnan(DeblurImgG[index])) DeblurImgG[index] = 0;
        if (std::isnan(DeblurImgB[index])) DeblurImgB[index] = 0;
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error R: %f\n",
           ComputeRMSError(GroundTruthImgR, DeblurImgR, width, height));
    printf("RMS Error G: %f\n",
           ComputeRMSError(GroundTruthImgG, DeblurImgG, width, height));
    printf("RMS Error B: %f\n",
           ComputeRMSError(GroundTruthImgB, DeblurImgB, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralReg(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);
  std::vector<float> BilateralRegImg(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImg, InputWeight, width, height,
                          BlurImgBuffer.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBuffer[index] > 0.001f) {
            DeltaImg[index] = BlurImg[index] / BlurImgBuffer[index];
          } else {
            DeltaImg[index] = BlurImg[index] / 0.001f;
          }
        } else {
          DeltaImg[index] = BlurImg[index] - BlurImgBuffer[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                          iheight, ErrorImgBuffer.data(),
                          ErrorWeightBuffer.data(), width, height, false);
    ComputeBilaterRegImage(DeblurImg, width, height, BilateralRegImg.data());

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImg[index] *=
              ErrorImgBuffer[index] / (1 + lambda * BilateralRegImg[index]);
        } else {
          DeblurImg[index] +=
              ErrorImgBuffer[index] - lambda * BilateralRegImg[index];
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error: %f\n",
           ComputeRMSError(GroundTruthImg, DeblurImg, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralReg(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgR(iwidth * iheight);
  std::vector<float> DeltaImgG(iwidth * iheight);
  std::vector<float> DeltaImgB(iwidth * iheight);
  std::vector<float> BilateralRegImgR(width * height);
  std::vector<float> BilateralRegImgG(width * height);
  std::vector<float> BilateralRegImgB(width * height);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImg(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                          width, height, BlurImgBufferR.data(),
                          BlurImgBufferG.data(), BlurImgBufferB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferR[index] > 0.001f) {
            DeltaImgR[index] = BlurImgR[index] / BlurImgBufferR[index];
          } else {
            DeltaImgR[index] = BlurImgR[index] / 0.001f;
          }
          if (BlurImgBufferG[index] > 0.001f) {
            DeltaImgG[index] = BlurImgG[index] / BlurImgBufferG[index];
          } else {
            DeltaImgG[index] = BlurImgG[index] / 0.001f;
          }
          if (BlurImgBufferB[index] > 0.001f) {
            DeltaImgB[index] = BlurImgB[index] / BlurImgBufferB[index];
          } else {
            DeltaImgB[index] = BlurImgB[index] / 0.001f;
          }
        } else {
          DeltaImgR[index] = BlurImgR[index] - BlurImgBufferR[index];
          DeltaImgG[index] = BlurImgG[index] - BlurImgBufferG[index];
          DeltaImgB[index] = BlurImgB[index] - BlurImgBufferB[index];
        }
      }
    }
    GenerateMotionBlurImg(DeltaImgR.data(), DeltaImgG.data(), DeltaImgB.data(),
                          BlurWeightBuffer.data(), iwidth, iheight,
                          ErrorImgBufferR.data(), ErrorImgBufferG.data(),
                          ErrorImgBufferB.data(), ErrorWeightBuffer.data(),
                          width, height, false);
    ComputeBilaterRegImage(DeblurImgR, width, height, BilateralRegImgR.data());
    ComputeBilaterRegImage(DeblurImgG, width, height, BilateralRegImgG.data());
    ComputeBilaterRegImage(DeblurImgB, width, height, BilateralRegImgB.data());

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgR[index] *=
              ErrorImgBufferR[index] / (1 + lambda * BilateralRegImgR[index]);
          DeblurImgG[index] *=
              ErrorImgBufferG[index] / (1 + lambda * BilateralRegImgG[index]);
          DeblurImgB[index] *=
              ErrorImgBufferB[index] / (1 + lambda * BilateralRegImgB[index]);
        } else {
          DeblurImgR[index] +=
              ErrorImgBufferR[index] - lambda * BilateralRegImgR[index];
          DeblurImgG[index] +=
              ErrorImgBufferG[index] - lambda * BilateralRegImgG[index];
          DeblurImgB[index] +=
              ErrorImgBufferB[index] - lambda * BilateralRegImgB[index];
        }

        DeblurImgR[index] = std::clamp(DeblurImgR[index], 0.0f, 1.0f);
        DeblurImgG[index] = std::clamp(DeblurImgG[index], 0.0f, 1.0f);
        DeblurImgB[index] = std::clamp(DeblurImgB[index], 0.0f, 1.0f);
      }
    }

#ifdef __SHOWERROR__
    printf("RMS Error R: %f\n",
           ComputeRMSError(GroundTruthImgR, DeblurImgR, width, height));
    printf("RMS Error G: %f\n",
           ComputeRMSError(GroundTruthImgG, DeblurImgG, width, height));
    printf("RMS Error B: %f\n",
           ComputeRMSError(GroundTruthImgB, DeblurImgB, width, height));
#else
    printf(".");
#endif
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralLapReg(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int i = 0, t = 1;
  // Parameters are set according to Levin et al Siggraph'07
  float powD = 0.8f, noiseVar = 0.0005f, epilson = t / 255.0f;
  float minWeight =
      exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

  // Bilateral Laplician Regularization
  for (i = 0; i <= t; i++) {
    BilateralTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    BilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                         pow(i / 255.0f, powD - 1.0f)) /
                        minWeight;
  }

  ProjectiveMotionRLDeblurBilateralReg(BlurImg, iwidth, iheight, DeblurImg,
                                       width, height, Niter, bPoisson, lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    BilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralLapReg(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int i = 0, t = 1;
  // Parameters are set according to Levin et al Siggraph'07
  float powD = 0.8f, noiseVar = 0.005f, epilson = t / 255.0f;
  float minWeight =
      exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

  // Bilateral Laplician Regularization
  for (i = 0; i <= t; i++) {
    BilateralTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    BilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                         pow(i / 255.0f, powD - 1.0f)) /
                        minWeight;
  }

  ProjectiveMotionRLDeblurBilateralReg(
      BlurImgR, BlurImgG, BlurImgB, iwidth, iheight, DeblurImgR, DeblurImgG,
      DeblurImgB, width, height, Niter, bPoisson, lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    BilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void ProjectiveMotionRL::ComputeGradientXImage(float* Img, int width,
                                               int height, float* DxImg,
                                               bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (x > 0) {
          DxImg[index] = Img[index] - Img[index - 1];
        } else {
          DxImg[index] = 0;
        }
      } else {
        if (x < width - 1) {
          DxImg[index] = Img[index] - Img[index + 1];
        } else {
          DxImg[index] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeGradientYImage(float* Img, int width,
                                               int height, float* DyImg,
                                               bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (y > 0) {
          DyImg[index] = Img[index] - Img[index - width];
        } else {
          DyImg[index] = 0;
        }
      } else {
        if (y < height - 1) {
          DyImg[index] = Img[index] - Img[index + width];
        } else {
          DyImg[index] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeGradientImage(float* Img, int width, int height,
                                              float* DxImg, float* DyImg,
                                              bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (x > 0) {
          DxImg[index] = Img[index] - Img[index - 1];
        } else {
          DxImg[index] = 0;
        }
        if (y > 0) {
          DyImg[index] = Img[index] - Img[index - width];
        } else {
          DyImg[index] = 0;
        }
      } else {
        if (x < width - 1) {
          DxImg[index] = Img[index] - Img[index + 1];
        } else {
          DxImg[index] = 0;
        }
        if (y < height - 1) {
          DyImg[index] = Img[index] - Img[index + width];
        } else {
          DyImg[index] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeBilaterRegImage(float* Img, int width,
                                                int height, float* BRImg) {
  // Sigma approximately equal to 1
  float GauFilter[5][5] = {{0.01f, 0.02f, 0.03f, 0.02f, 0.01f},
                           {0.02f, 0.03f, 0.04f, 0.03f, 0.02f},
                           {0.03f, 0.04f, 0.05f, 0.04f, 0.03f},
                           {0.02f, 0.03f, 0.04f, 0.03f, 0.02f},
                           {0.01f, 0.02f, 0.03f, 0.02f, 0.01f}};
  int x = 0, y = 0, index = 0, xx = 0, yy = 0, iindex = 0, iiindex = 0;
  memset(BRImg, 0, width * height * sizeof(float));

  // Compute the long distance 2nd derivative image weighted by Bilateral filter
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      for (xx = -2; xx <= 2; xx++) {
        if (x + xx >= 0 && x + xx < width && x - xx >= 0 && x - xx < width) {
          for (yy = -2; yy <= 2; yy++) {
            if (y + yy >= 0 && y + yy < height && y - yy >= 0 &&
                y - yy < height) {
              iindex = (y + yy) * width + (x + xx);
              iiindex = (y - yy) * width + (x - xx);
              BRImg[index] +=
                  GauFilter[xx + 2][yy + 2] * 0.5f *
                  (BilateralTable[(int)(fabs(Img[iindex] - Img[index]) *
                                        255.0f)] +
                   BilateralTable[(int)(fabs(Img[iiindex] - Img[index]) *
                                        255.0f)]) *
                  (2 * Img[index] - Img[iindex] - Img[iiindex]);
            }
          }
        }
      }
    }
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurMultiScale(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, int Nscale, bool bPoisson) {
  int i = 0, iscale = 0, x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;
  float HFactor[NumSamples];
  float ScaleFactor = sqrt(2.0f);
  for (i = 0; i < NumSamples; i++) {
    HFactor[i] = Hmatrix[i].Hmatrix[2][2];
  }

  std::vector<float> DeltaImg(width * height);

  ClearBuffer();
  SetBuffer(width, height);

  for (iscale = Nscale - 1; iscale >= 0; iscale--) {
    float powfactor = pow(ScaleFactor, iscale);
    int bwidth = (int)(iwidth / powfactor),
        bheight = (int)(iheight / powfactor);
    printf("Level %d: %d %d\n", iscale, bwidth, bheight);
    float bfactw = (float)(iwidth - 1) / (float)(bwidth - 1),
          bfacth = (float)(iheight - 1) / (float)(bheight - 1);
    for (i = 0; i < NumSamples; i++) {
      Hmatrix[i].Hmatrix[2][2] = HFactor[i] * powfactor;
      Hmatrix[i].MatrixInverse(Hmatrix[i].Hmatrix, IHmatrix[i].Hmatrix);
    }

    for (itr = 0; itr < Niter; itr++) {
      printf("%d\n", itr);
      GenerateMotionBlurImg(DeblurImg, InputWeight, width, height,
                            BlurImgBuffer.data(), BlurWeightBuffer.data(),
                            bwidth, bheight, true);
      for (y = 0, index = 0; y < bheight; y++) {
        for (x = 0; x < bwidth; x++, index++) {
          float bvalue = ReturnInterpolatedValueFast(x * bfactw, y * bfacth,
                                                     BlurImg, iwidth, iheight);
          if (bPoisson) {
            if (BlurImgBuffer[index] > 0.0001f) {
              DeltaImg[index] = bvalue / BlurImgBuffer[index];
            } else {
              DeltaImg[index] = bvalue / 0.0001f;
            }
          } else {
            DeltaImg[index] = bvalue - BlurImgBuffer[index];
          }
        }
      }
      GenerateMotionBlurImg(DeltaImg.data(), BlurWeightBuffer.data(), bwidth,
                            bheight, ErrorImgBuffer.data(),
                            ErrorWeightBuffer.data(), width, height, false);
      for (y = 0, index = 0; y < height; y++) {
        for (x = 0; x < width; x++, index++) {
          if (bPoisson) {
            DeblurImg[index] *= ErrorImgBuffer[index];
          } else {
            DeblurImg[index] += ErrorImgBuffer[index];
          }
          DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
        }
      }
    }
  }
}
