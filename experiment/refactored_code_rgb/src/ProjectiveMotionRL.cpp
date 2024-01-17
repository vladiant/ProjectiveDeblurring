#include "ProjectiveMotionRL.h"

#include <algorithm>
#include <cmath>
#include <memory>

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

void ProjectiveMotionRL::WarpImageGray(float* InputImg, float* inputWeight,
                                       int iwidth, int iheight,
                                       float* OutputImg, float* outputWeight,
                                       int width, int height, int i) {
  if (i >= 0 && i < NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, Hmatrix[-i]);
  }
}

void ProjectiveMotionRL::WarpImageRgb(float* InputImgRGB, float* inputWeight,
                                      int iwidth, int iheight,
                                      float* OutputImgRGB, float* outputWeight,
                                      int width, int height, int i) {
  if (i >= 0 && i < NumSamples) {
    warpImageRgb(InputImgRGB, inputWeight, iwidth, iheight, OutputImgRGB,
                 outputWeight, width, height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImageRgb(InputImgRGB, inputWeight, iwidth, iheight, OutputImgRGB,
                 outputWeight, width, height, Hmatrix[-i]);
  }
}

void ProjectiveMotionRL::GenerateMotionBlurImgGray(
    float* InputImg, float* inputWeight, int iwidth, int iheight,
    float* BlurImg, float* outputWeight, int width, int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  if (WarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImg, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImageGray(InputImg, inputWeight, iwidth, iheight,
                    WarpImgBuffer.data(), WarpWeightBuffer.data(), width,
                    height, i);
    } else {
      WarpImageGray(InputImg, inputWeight, iwidth, iheight,
                    WarpImgBuffer.data(), WarpWeightBuffer.data(), width,
                    height, -i);
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

void ProjectiveMotionRL::GenerateMotionBlurImgRgb(
    float* InputImgRGB, float* inputWeight, int iwidth, int iheight,
    float* BlurImgRGB, float* outputWeight, int width, int height,
    bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  if (WarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImgRGB, 0, totalpixel * sizeof(float) * 3);
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImageRgb(InputImgRGB, inputWeight, iwidth, iheight,
                   WarpImgBufferRGB.data(), WarpWeightBuffer.data(), width,
                   height, i);
    } else {
      WarpImageRgb(InputImgRGB, inputWeight, iwidth, iheight,
                   WarpImgBufferRGB.data(), WarpWeightBuffer.data(), width,
                   height, -i);
    }

    for (index = 0; index < totalpixel; index++) {
      BlurImgRGB[3 * index] +=
          WarpImgBufferRGB[3 * index] * WarpWeightBuffer[index];
      BlurImgRGB[3 * index + 1] +=
          WarpImgBufferRGB[3 * index + 1] * WarpWeightBuffer[index];
      BlurImgRGB[3 * index + 2] +=
          WarpImgBufferRGB[3 * index + 2] * WarpWeightBuffer[index];
      outputWeight[index] += WarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImgRGB[3 * index] /= outputWeight[index];
    BlurImgRGB[3 * index + 1] /= outputWeight[index];
    BlurImgRGB[3 * index + 2] /= outputWeight[index];
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurGray(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
                              BlurImgBuffer.data(), BlurWeightBuffer.data(),
                              iwidth, iheight, true);
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
    GenerateMotionBlurImgGray(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurRgb(float* BlurImgRGB,
                                                     int iwidth, int iheight,
                                                     float* DeblurImgRGB,
                                                     int width, int height,
                                                     int Niter, bool bPoisson) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgRGB(iwidth * iheight * 3);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImgRgb(DeblurImgRGB, InputWeight, width, height,
                             BlurImgBufferRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferRGB[3 * index] > 0.001f) {
            DeltaImgRGB[3 * index] =
                BlurImgRGB[3 * index] / BlurImgBufferRGB[3 * index];
          } else {
            DeltaImgRGB[3 * index] = BlurImgRGB[3 * index] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 1] > 0.001f) {
            DeltaImgRGB[3 * index + 1] =
                BlurImgRGB[3 * index + 1] / BlurImgBufferRGB[3 * index + 1];
          } else {
            DeltaImgRGB[3 * index + 1] = BlurImgRGB[3 * index + 1] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 2] > 0.001f) {
            DeltaImgRGB[3 * index + 2] =
                BlurImgRGB[3 * index + 2] / BlurImgBufferRGB[3 * index + 2];
          } else {
            DeltaImgRGB[3 * index + 2] = BlurImgRGB[3 * index + 2] / 0.001f;
          }
        } else {
          DeltaImgRGB[3 * index] =
              BlurImgRGB[3 * index] - BlurImgBufferRGB[3 * index];
          DeltaImgRGB[3 * index + 1] =
              BlurImgRGB[3 * index + 1] - BlurImgBufferRGB[3 * index + 1];
          DeltaImgRGB[3 * index + 2] =
              BlurImgRGB[3 * index + 2] - BlurImgBufferRGB[3 * index + 2];
        }
      }
    }
    GenerateMotionBlurImgRgb(DeltaImgRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, ErrorImgBufferRGB.data(),
                             ErrorWeightBuffer.data(), width, height, false);
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgRGB[3 * index] *= ErrorImgBufferRGB[3 * index];
          DeblurImgRGB[3 * index + 1] *= ErrorImgBufferRGB[3 * index + 1];
          DeblurImgRGB[3 * index + 2] *= ErrorImgBufferRGB[3 * index + 2];
        } else {
          DeblurImgRGB[3 * index] += ErrorImgBufferRGB[3 * index];
          DeblurImgRGB[3 * index + 1] += ErrorImgBufferRGB[3 * index + 1];
          DeblurImgRGB[3 * index + 2] += ErrorImgBufferRGB[3 * index + 2];
        }

        DeblurImgRGB[3 * index] =
            std::clamp(DeblurImgRGB[3 * index], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 1] =
            std::clamp(DeblurImgRGB[3 * index + 1], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 2] =
            std::clamp(DeblurImgRGB[3 * index + 2], 0.0f, 1.0f);
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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurTVRegGray(
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
    GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
                              BlurImgBuffer.data(), BlurWeightBuffer.data(),
                              iwidth, iheight, true);
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
    GenerateMotionBlurImgGray(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                              iheight, ErrorImgBuffer.data(),
                              ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImageGray(DeblurImg, width, height, DxImg.data(),
                             DyImg.data(), true);
    for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
      for (x = 0; x < width; x++, index++) {
        if (DxImg[index] > 0) DxImg[index] = 1.0f / 255.0f;
        if (DxImg[index] < 0) DxImg[index] = -1.0f / 255.0f;
        if (DyImg[index] > 0) DyImg[index] = 1.0f / 255.0f;
        if (DyImg[index] < 0) DyImg[index] = -1.0f / 255.0f;
      }
    }
    ComputeGradientXImageGray(DxImg.data(), width, height, DxxImg.data(),
                              false);
    ComputeGradientYImageGray(DyImg.data(), width, height, DyyImg.data(),
                              false);

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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurTVRegRgb(
    float* BlurImgRGB, int iwidth, int iheight, float* DeblurImgRGB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgRGB(iwidth * iheight * 3);

  std::vector<float> DxImgRGB(width * height * 3);
  std::vector<float> DyImgRGB(width * height * 3);
  std::vector<float> DxxImgRGB(width * height * 3);
  std::vector<float> DyyImgRGB(width * height * 3);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImgRgb(DeblurImgRGB, InputWeight, width, height,
                             BlurImgBufferRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferRGB[3 * index] > 0.001f) {
            DeltaImgRGB[3 * index] =
                BlurImgRGB[3 * index] / BlurImgBufferRGB[3 * index];
          } else {
            DeltaImgRGB[3 * index] = BlurImgRGB[3 * index] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 1] > 0.001f) {
            DeltaImgRGB[3 * index + 1] =
                BlurImgRGB[3 * index + 1] / BlurImgBufferRGB[3 * index + 1];
          } else {
            DeltaImgRGB[3 * index + 1] = BlurImgRGB[3 * index + 1] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 2] > 0.001f) {
            DeltaImgRGB[3 * index + 2] =
                BlurImgRGB[3 * index + 2] / BlurImgBufferRGB[3 * index + 2];
          } else {
            DeltaImgRGB[3 * index + 2] = BlurImgRGB[3 * index + 2] / 0.001f;
          }
        } else {
          DeltaImgRGB[3 * index] =
              BlurImgRGB[3 * index] - BlurImgBufferRGB[3 * index];
          DeltaImgRGB[3 * index + 1] =
              BlurImgRGB[3 * index + 1] - BlurImgBufferRGB[3 * index + 1];
          DeltaImgRGB[3 * index + 2] =
              BlurImgRGB[3 * index + 2] - BlurImgBufferRGB[3 * index + 2];
        }
      }
    }
    GenerateMotionBlurImgRgb(DeltaImgRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, ErrorImgBufferRGB.data(),
                             ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImageRgb(DeblurImgRGB, width, height, DxImgRGB.data(),
                            DyImgRGB.data(), true);
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (DxImgRGB[3 * index] > 0) DxImgRGB[3 * index] = 1.0f / 255.0f;
        if (DxImgRGB[3 * index] < 0) DxImgRGB[3 * index] = -1.0f / 255.0f;
        if (DyImgRGB[3 * index] > 0) DyImgRGB[3 * index] = 1.0f / 255.0f;
        if (DyImgRGB[3 * index] < 0) DyImgRGB[3 * index] = -1.0f / 255.0f;
        if (DxImgRGB[3 * index + 1] > 0)
          DxImgRGB[3 * index + 1] = 1.0f / 255.0f;
        if (DxImgRGB[3 * index + 1] < 0)
          DxImgRGB[3 * index + 1] = -1.0f / 255.0f;
        if (DyImgRGB[3 * index + 1] > 0)
          DyImgRGB[3 * index + 1] = 1.0f / 255.0f;
        if (DyImgRGB[3 * index + 1] < 0)
          DyImgRGB[3 * index + 1] = -1.0f / 255.0f;
        if (DxImgRGB[3 * index + 2] > 0)
          DxImgRGB[3 * index + 2] = 1.0f / 255.0f;
        if (DxImgRGB[3 * index + 2] < 0)
          DxImgRGB[3 * index + 2] = -1.0f / 255.0f;
        if (DyImgRGB[3 * index + 2] > 0)
          DyImgRGB[3 * index + 2] = 1.0f / 255.0f;
        if (DyImgRGB[3 * index + 2] < 0)
          DyImgRGB[3 * index + 2] = -1.0f / 255.0f;
      }
    }
    ComputeGradientXImageRgb(DxImgRGB.data(), width, height, DxxImgRGB.data(),
                             false);
    ComputeGradientYImageRgb(DyImgRGB.data(), width, height, DyyImgRGB.data(),
                             false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgRGB[3 * index] *=
              ErrorImgBufferRGB[3 * index] /
              (1 + lambda * (DxxImgRGB[3 * index] + DyyImgRGB[3 * index]));
          DeblurImgRGB[3 * index + 1] *=
              ErrorImgBufferRGB[3 * index + 1] /
              (1 +
               lambda * (DxxImgRGB[3 * index + 1] + DyyImgRGB[3 * index + 1]));
          DeblurImgRGB[3 * index + 2] *=
              ErrorImgBufferRGB[3 * index + 2] /
              (1 +
               lambda * (DxxImgRGB[3 * index + 2] + DyyImgRGB[3 * index + 2]));
        } else {
          DeblurImgRGB[3 * index] +=
              ErrorImgBufferRGB[3 * index] -
              lambda * (DxxImgRGB[3 * index] + DyyImgRGB[3 * index]);
          DeblurImgRGB[3 * index + 1] +=
              ErrorImgBufferRGB[3 * index + 1] -
              lambda * (DxxImgRGB[3 * index + 1] + DyyImgRGB[3 * index + 1]);
          DeblurImgRGB[3 * index + 2] +=
              ErrorImgBufferRGB[3 * index + 2] -
              lambda * (DxxImgRGB[3 * index + 2] + DyyImgRGB[3 * index + 2]);
        }
        DeblurImgRGB[3 * index] =
            std::clamp(DeblurImgRGB[3 * index], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 1] =
            std::clamp(DeblurImgRGB[3 * index + 1], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 2] =
            std::clamp(DeblurImgRGB[3 * index + 2], 0.0f, 1.0f);
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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurSpsRegGray(
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
    GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
                              BlurImgBuffer.data(), BlurWeightBuffer.data(),
                              iwidth, iheight, true);
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
    GenerateMotionBlurImgGray(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                              iheight, ErrorImgBuffer.data(),
                              ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImageGray(DeblurImg, width, height, DxImg.data(),
                             DyImg.data(), true);

    ComputeGradientXImageGray(DxImg.data(), width, height, DxxImg.data(),
                              false);
    ComputeGradientYImageGray(DyImg.data(), width, height, DyyImg.data(),
                              false);

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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurSpsRegRgb(
    float* BlurImgRGB, int iwidth, int iheight, float* DeblurImgRGB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;
  float WxR = NAN, WyR = NAN, WxG = NAN, WyG = NAN, WxB = NAN, WyB = NAN;

  std::vector<float> DeltaImgRGB(iwidth * iheight * 3);

  std::vector<float> DxImgRGB(width * height * 3);
  std::vector<float> DyImgRGB(width * height * 3);
  std::vector<float> DxxImgRGB(width * height * 3);
  std::vector<float> DyyImgRGB(width * height * 3);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImgRgb(DeblurImgRGB, InputWeight, width, height,
                             BlurImgBufferRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, true);

    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferRGB[3 * index] > 0.001f) {
            DeltaImgRGB[3 * index] =
                BlurImgRGB[3 * index] / BlurImgBufferRGB[3 * index];
          } else {
            DeltaImgRGB[3 * index] = BlurImgRGB[3 * index] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 1] > 0.001f) {
            DeltaImgRGB[3 * index + 1] =
                BlurImgRGB[3 * index + 1] / BlurImgBufferRGB[3 * index + 1];
          } else {
            DeltaImgRGB[3 * index + 1] = BlurImgRGB[3 * index + 1] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 2] > 0.001f) {
            DeltaImgRGB[3 * index + 2] =
                BlurImgRGB[3 * index + 2] / BlurImgBufferRGB[3 * index + 2];
          } else {
            DeltaImgRGB[3 * index + 2] = BlurImgRGB[3 * index + 2] / 0.001f;
          }
        } else {
          DeltaImgRGB[3 * index] =
              BlurImgRGB[3 * index] - BlurImgBufferRGB[3 * index];
          DeltaImgRGB[3 * index + 1] =
              BlurImgRGB[3 * index + 1] - BlurImgBufferRGB[3 * index + 1];
          DeltaImgRGB[3 * index + 2] =
              BlurImgRGB[3 * index + 2] - BlurImgBufferRGB[3 * index + 2];
        }
      }
    }
    GenerateMotionBlurImgRgb(DeltaImgRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, ErrorImgBufferRGB.data(),
                             ErrorWeightBuffer.data(), width, height, false);

    ComputeGradientImageRgb(DeblurImgRGB, width, height, DxImgRGB.data(),
                            DyImgRGB.data(), true);

    ComputeGradientXImageRgb(DxImgRGB.data(), width, height, DxxImgRGB.data(),
                             false);
    ComputeGradientYImageRgb(DyImgRGB.data(), width, height, DyyImgRGB.data(),
                             false);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        WxR = getSpsWeight(DxImgRGB[3 * index]);
        WyR = getSpsWeight(DyImgRGB[3 * index]);
        WxG = getSpsWeight(DxImgRGB[3 * index + 1]);
        WyG = getSpsWeight(DyImgRGB[3 * index + 1]);
        WxB = getSpsWeight(DxImgRGB[3 * index + 2]);
        WyB = getSpsWeight(DyImgRGB[3 * index + 2]);

        if (bPoisson) {
          DeblurImgRGB[3 * index] *=
              ErrorImgBufferRGB[3 * index] /
              (1 + lambda * (WxR * DxxImgRGB[3 * index] +
                             WyR * DyyImgRGB[3 * index]));
          DeblurImgRGB[3 * index + 1] *=
              ErrorImgBufferRGB[3 * index + 1] /
              (1 + lambda * (WxG * DxxImgRGB[3 * index + 1] +
                             WyG * DyyImgRGB[3 * index + 1]));
          DeblurImgRGB[3 * index + 2] *=
              ErrorImgBufferRGB[3 * index + 2] /
              (1 + lambda * (WxB * DxxImgRGB[3 * index + 2] +
                             WyB * DyyImgRGB[3 * index + 2]));
        } else {
          DeblurImgRGB[3 * index] += ErrorImgBufferRGB[3 * index] -
                                     lambda * (WxR * DxxImgRGB[3 * index] +
                                               WyR * DyyImgRGB[3 * index]);
          DeblurImgRGB[3 * index + 1] +=
              ErrorImgBufferRGB[3 * index + 1] -
              lambda * (WxG * DxxImgRGB[3 * index + 1] +
                        WyG * DyyImgRGB[3 * index + 1]);
          DeblurImgRGB[3 * index + 2] +=
              ErrorImgBufferRGB[3 * index + 2] -
              lambda * (WxB * DxxImgRGB[3 * index + 2] +
                        WyB * DyyImgRGB[3 * index + 2]);
        }
        DeblurImgRGB[3 * index] =
            std::clamp(DeblurImgRGB[3 * index], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 1] =
            std::clamp(DeblurImgRGB[3 * index + 1], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 2] =
            std::clamp(DeblurImgRGB[3 * index + 2], 0.0f, 1.0f);

        if (std::isnan(DeblurImgRGB[3 * index])) DeblurImgRGB[3 * index] = 0;
        if (std::isnan(DeblurImgRGB[3 * index + 1]))
          DeblurImgRGB[3 * index + 1] = 0;
        if (std::isnan(DeblurImgRGB[3 * index + 2]))
          DeblurImgRGB[3 * index + 2] = 0;
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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralRegGray(
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
    GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
                              BlurImgBuffer.data(), BlurWeightBuffer.data(),
                              iwidth, iheight, true);
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
    GenerateMotionBlurImgGray(DeltaImg.data(), BlurWeightBuffer.data(), iwidth,
                              iheight, ErrorImgBuffer.data(),
                              ErrorWeightBuffer.data(), width, height, false);
    ComputeBilaterRegImageGray(DeblurImg, width, height,
                               BilateralRegImg.data());

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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralRegRgb(
    float* BlurImgRGB, int iwidth, int iheight, float* DeblurImgRGB, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImgRGB(iwidth * iheight * 3);
  std::vector<float> BilateralRegImgRGB(width * height * 3);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < Niter; itr++) {
    GenerateMotionBlurImgRgb(DeblurImgRGB, InputWeight, width, height,
                             BlurImgBufferRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (BlurImgBufferRGB[3 * index] > 0.001f) {
            DeltaImgRGB[3 * index] =
                BlurImgRGB[3 * index] / BlurImgBufferRGB[3 * index];
          } else {
            DeltaImgRGB[3 * index] = BlurImgRGB[3 * index] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 1] > 0.001f) {
            DeltaImgRGB[3 * index + 1] =
                BlurImgRGB[3 * index + 1] / BlurImgBufferRGB[3 * index + 1];
          } else {
            DeltaImgRGB[3 * index + 1] = BlurImgRGB[3 * index + 1] / 0.001f;
          }
          if (BlurImgBufferRGB[3 * index + 2] > 0.001f) {
            DeltaImgRGB[3 * index + 2] =
                BlurImgRGB[3 * index + 2] / BlurImgBufferRGB[3 * index + 2];
          } else {
            DeltaImgRGB[3 * index + 2] = BlurImgRGB[3 * index + 2] / 0.001f;
          }
        } else {
          DeltaImgRGB[3 * index] =
              BlurImgRGB[3 * index] - BlurImgBufferRGB[3 * index];
          DeltaImgRGB[3 * index + 1] =
              BlurImgRGB[3 * index + 1] - BlurImgBufferRGB[3 * index + 1];
          DeltaImgRGB[3 * index + 2] =
              BlurImgRGB[3 * index + 2] - BlurImgBufferRGB[3 * index + 2];
        }
      }
    }
    GenerateMotionBlurImgRgb(DeltaImgRGB.data(), BlurWeightBuffer.data(),
                             iwidth, iheight, ErrorImgBufferRGB.data(),
                             ErrorWeightBuffer.data(), width, height, false);
    ComputeBilaterRegImageRgb(DeblurImgRGB, width, height,
                              BilateralRegImgRGB.data());

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgRGB[3 * index] *=
              ErrorImgBufferRGB[3 * index] /
              (1 + lambda * BilateralRegImgRGB[3 * index]);
          DeblurImgRGB[3 * index + 1] *=
              ErrorImgBufferRGB[3 * index + 1] /
              (1 + lambda * BilateralRegImgRGB[3 * index + 1]);
          DeblurImgRGB[3 * index] *=
              ErrorImgBufferRGB[3 * index] /
              (1 + lambda * BilateralRegImgRGB[3 * index]);
        } else {
          DeblurImgRGB[3 * index] += ErrorImgBufferRGB[3 * index] -
                                     lambda * BilateralRegImgRGB[3 * index];
          DeblurImgRGB[3 * index + 1] +=
              ErrorImgBufferRGB[3 * index + 1] -
              lambda * BilateralRegImgRGB[3 * index + 1];
          DeblurImgRGB[index + 2] += ErrorImgBufferRGB[3 * index + 2] -
                                     lambda * BilateralRegImgRGB[3 * index + 2];
        }

        DeblurImgRGB[3 * index] =
            std::clamp(DeblurImgRGB[3 * index], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 1] =
            std::clamp(DeblurImgRGB[3 * index + 1], 0.0f, 1.0f);
        DeblurImgRGB[3 * index + 2] =
            std::clamp(DeblurImgRGB[3 * index + 2], 0.0f, 1.0f);
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

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralLapRegGray(
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

  ProjectiveMotionRLDeblurBilateralRegGray(BlurImg, iwidth, iheight, DeblurImg,
                                           width, height, Niter, bPoisson,
                                           lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    BilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurBilateralLapRegRgb(
    float* BlurImgRGB, int iwidth, int iheight, float* DeblurImgRGB, int width,
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

  ProjectiveMotionRLDeblurBilateralRegRgb(BlurImgRGB, iwidth, iheight,
                                          DeblurImgRGB, width, height, Niter,
                                          bPoisson, lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    BilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void ProjectiveMotionRL::ComputeGradientXImageGray(float* Img, int width,
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

void ProjectiveMotionRL::ComputeGradientXImageRgb(float* ImgRGB, int width,
                                                  int height, float* DxImgRGB,
                                                  bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (x > 0) {
          DxImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index - 1)];
          DxImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index - 1) + 1];
          DxImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index - 1) + 2];
        } else {
          DxImgRGB[3 * index] = 0;
          DxImgRGB[3 * index + 1] = 0;
          DxImgRGB[3 * index + 2] = 0;
        }
      } else {
        if (x < width - 1) {
          DxImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index + 1)];
          DxImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index + 1) + 1];
          DxImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index + 1) + 2];
        } else {
          DxImgRGB[3 * index] = 0;
          DxImgRGB[3 * index + 1] = 0;
          DxImgRGB[3 * index + 2] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeGradientYImageGray(float* Img, int width,
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

void ProjectiveMotionRL::ComputeGradientYImageRgb(float* ImgRGB, int width,
                                                  int height, float* DyImgRGB,
                                                  bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (y > 0) {
          DyImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index - width)];
          DyImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index - width) + 1];
          DyImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index - width) + 2];
        } else {
          DyImgRGB[3 * index] = 0;
          DyImgRGB[3 * index + 1] = 0;
          DyImgRGB[3 * index + 2] = 0;
        }
      } else {
        if (y < height - 1) {
          DyImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index + width)];
          DyImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index + width) + 1];
          DyImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index + width) + 2];
        } else {
          DyImgRGB[3 * index] = 0;
          DyImgRGB[3 * index + 1] = 0;
          DyImgRGB[3 * index + 2] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeGradientImageGray(float* Img, int width,
                                                  int height, float* DxImg,
                                                  float* DyImg, bool bflag) {
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

void ProjectiveMotionRL::ComputeGradientImageRgb(float* ImgRGB, int width,
                                                 int height, float* DxImgRGB,
                                                 float* DyImgRGB, bool bflag) {
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bflag) {
        if (x > 0) {
          DxImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index - 1)];
          DxImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index - 1) + 1];
          DxImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index - 1) + 2];
        } else {
          DxImgRGB[3 * index] = 0;
          DxImgRGB[3 * index + 1] = 0;
          DxImgRGB[3 * index + 2] = 0;
        }
        if (y > 0) {
          DyImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index - width)];
          DyImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index - width) + 1];
          DyImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index - width) + 2];
        } else {
          DyImgRGB[3 * index] = 0;
          DyImgRGB[3 * index + 1] = 0;
          DyImgRGB[3 * index + 2] = 0;
        }
      } else {
        if (x < width - 1) {
          DxImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index + 1)];
          DxImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index + 1) + 1];
          DxImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index + 1) + 2];
        } else {
          DxImgRGB[3 * index] = 0;
          DxImgRGB[3 * index + 1] = 0;
          DxImgRGB[3 * index + 2] = 0;
        }
        if (y < height - 1) {
          DyImgRGB[3 * index] = ImgRGB[3 * index] - ImgRGB[3 * (index + width)];
          DyImgRGB[3 * index + 1] =
              ImgRGB[3 * index + 1] - ImgRGB[3 * (index + width) + 1];
          DyImgRGB[3 * index + 2] =
              ImgRGB[3 * index + 2] - ImgRGB[3 * (index + width) + 2];
        } else {
          DyImgRGB[3 * index] = 0;
          DyImgRGB[3 * index + 1] = 0;
          DyImgRGB[3 * index + 2] = 0;
        }
      }
    }
  }
}

void ProjectiveMotionRL::ComputeBilaterRegImageGray(float* Img, int width,
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

void ProjectiveMotionRL::ComputeBilaterRegImageRgb(float* ImgRGB, int width,
                                                   int height,
                                                   float* BRImgRGB) {
  // Sigma approximately equal to 1
  float GauFilter[5][5] = {{0.01f, 0.02f, 0.03f, 0.02f, 0.01f},
                           {0.02f, 0.03f, 0.04f, 0.03f, 0.02f},
                           {0.03f, 0.04f, 0.05f, 0.04f, 0.03f},
                           {0.02f, 0.03f, 0.04f, 0.03f, 0.02f},
                           {0.01f, 0.02f, 0.03f, 0.02f, 0.01f}};
  int x = 0, y = 0, index = 0, xx = 0, yy = 0, iindex = 0, iiindex = 0;
  memset(BRImgRGB, 0, width * height * sizeof(float) * 3);

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
              BRImgRGB[3 * index] +=
                  GauFilter[xx + 2][yy + 2] * 0.5f *
                  (BilateralTable[(
                       int)(fabs(ImgRGB[3 * iindex] - ImgRGB[3 * index]) *
                            255.0f)] +
                   BilateralTable[(
                       int)(fabs(ImgRGB[3 * iiindex] - ImgRGB[3 * index]) *
                            255.0f)]) *
                  (2 * ImgRGB[3 * index] - ImgRGB[3 * iindex] -
                   ImgRGB[3 * iiindex]);

              BRImgRGB[3 * index + 1] +=
                  GauFilter[xx + 2][yy + 2] * 0.5f *
                  (BilateralTable[(int)(fabs(ImgRGB[3 * iindex + 1] -
                                             ImgRGB[3 * index + 1]) *
                                        255.0f)] +
                   BilateralTable[(int)(fabs(ImgRGB[3 * iiindex + 1] -
                                             ImgRGB[3 * index + 1]) *
                                        255.0f)]) *
                  (2 * ImgRGB[3 * index + 1] - ImgRGB[3 * iindex + 1] -
                   ImgRGB[3 * iiindex + 1]);

              BRImgRGB[3 * index + 2] +=
                  GauFilter[xx + 2][yy + 2] * 0.5f *
                  (BilateralTable[(int)(fabs(ImgRGB[3 * iindex + 2] -
                                             ImgRGB[3 * index + 2]) *
                                        255.0f)] +
                   BilateralTable[(int)(fabs(ImgRGB[3 * iiindex + 2] -
                                             ImgRGB[3 * index + 2]) *
                                        255.0f)]) *
                  (2 * ImgRGB[3 * index + 2] - ImgRGB[3 * iindex + 2] -
                   ImgRGB[3 * iiindex + 2]);
            }
          }
        }
      }
    }
  }
}

void ProjectiveMotionRL::ProjectiveMotionRLDeblurMultiScaleGray(
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
      GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
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
      GenerateMotionBlurImgGray(DeltaImg.data(), BlurWeightBuffer.data(),
                                bwidth, bheight, ErrorImgBuffer.data(),
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

float ProjectiveMotionRL::normalrand() {
  std::normal_distribution<float> normalDist(0.0f, 1.0f);
  // float val = 0;
  // for (int i = 0; i != 12; ++i) val += ((float)(rand()) / RAND_MAX);
  // return val - 6.0f;
  return normalDist(mRandomEngine);
}

void ProjectiveMotionRL::gaussianNoiseGray(float* Img, int width, int height,
                                           float amp) {
  int x, y, index;
  float random, noise;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      random = normalrand() / 255.0f;
      noise = amp * random;
      Img[index] += noise;
      std::clamp(Img[index], 0.0f, 1.0f);
    }
  }
}

void ProjectiveMotionRL::gaussianNoiseRGB(float* ImgRGB, int width, int height,
                                          float amp) {
  int x, y, index;
  float random, noise;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      random = normalrand() / 255.0f;
      noise = amp * random;
      ImgRGB[3 * index] += noise;
      std::clamp(ImgRGB[3 * index], 0.0f, 1.0f);
      random = normalrand() / 255.0f;
      noise = amp * random;
      ImgRGB[3 * index + 1] += noise;
      std::clamp(ImgRGB[3 * index + 1], 0.0f, 1.0f);
      random = normalrand() / 255.0f;
      noise = amp * random;
      ImgRGB[3 * index + 2] += noise;
      std::clamp(ImgRGB[3 * index + 2], 0.0f, 1.0f);
    }
  }
}