#include "RLDeblurrerBilateralLaplReg.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "BicubicInterpolation.h"
#include "ImResize.h"
#include "bitmap.h"
#include "warping.h"

RLDeblurrerBilateralLaplReg::RLDeblurrerBilateralLaplReg(
    IBlurImageGenerator& aBlurGenerator, IErrorCalculator& aErrorCalculator)
    : mBlurGenerator(aBlurGenerator), mErrorCalculator(aErrorCalculator) {
  SetBilateralTable();
}

void RLDeblurrerBilateralLaplReg::SetBuffer(int width, int height) {
  mBlurGenerator.SetBuffer(width, height);

  mBlurImgBuffer.resize(width * height);
  mBlurImgBufferR.resize(width * height);
  mBlurImgBufferG.resize(width * height);
  mBlurImgBufferB.resize(width * height);
  mBlurWeightBuffer.resize(width * height);
  mErrorImgBuffer.resize(width * height);
  mErrorImgBufferR.resize(width * height);
  mErrorImgBufferG.resize(width * height);
  mErrorImgBufferB.resize(width * height);
  mErrorWeightBuffer.resize(width * height);
}

void RLDeblurrerBilateralLaplReg::ClearBuffer() {
  mBlurGenerator.ClearBuffer();

  mBlurImgBuffer.clear();
  mBlurImgBufferR.clear();
  mBlurImgBufferG.clear();
  mBlurImgBufferB.clear();

  mBlurWeightBuffer.clear();

  mErrorImgBuffer.clear();
  mErrorImgBufferR.clear();
  mErrorImgBufferG.clear();
  mErrorImgBufferB.clear();

  mErrorWeightBuffer.clear();
}

void RLDeblurrerBilateralLaplReg::ProcessGray(float* BlurImg, int iwidth,
                                              int iheight, float* DeblurImg,
                                              int width, int height, int Niter,
                                              bool bPoisson, float lambda) {
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
    mBlurGenerator.blurGray(DeblurImg, InputWeight, width, height,
                            mBlurImgBuffer.data(), mBlurWeightBuffer.data(),
                            iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (mBlurImgBuffer[index] > 0.001f) {
            DeltaImg[index] = BlurImg[index] / mBlurImgBuffer[index];
          } else {
            DeltaImg[index] = BlurImg[index] / 0.001f;
          }
        } else {
          DeltaImg[index] = BlurImg[index] - mBlurImgBuffer[index];
        }
      }
    }
    mBlurGenerator.blurGray(DeltaImg.data(), mBlurWeightBuffer.data(), iwidth,
                            iheight, mErrorImgBuffer.data(),
                            mErrorWeightBuffer.data(), width, height, false);
    ComputeBilaterRegImageGray(DeblurImg, width, height,
                               BilateralRegImg.data());

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImg[index] *=
              mErrorImgBuffer[index] / (1 + lambda * BilateralRegImg[index]);
        } else {
          DeblurImg[index] +=
              mErrorImgBuffer[index] - lambda * BilateralRegImg[index];
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

    mErrorCalculator.calculateErrorGray(DeblurImg, width, height);
  }
}

void RLDeblurrerBilateralLaplReg::ProcessRgb(
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
    mBlurGenerator.blurRgb(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                           width, height, mBlurImgBufferR.data(),
                           mBlurImgBufferG.data(), mBlurImgBufferB.data(),
                           mBlurWeightBuffer.data(), iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (bPoisson) {
          if (mBlurImgBufferR[index] > 0.001f) {
            DeltaImgR[index] = BlurImgR[index] / mBlurImgBufferR[index];
          } else {
            DeltaImgR[index] = BlurImgR[index] / 0.001f;
          }
          if (mBlurImgBufferG[index] > 0.001f) {
            DeltaImgG[index] = BlurImgG[index] / mBlurImgBufferG[index];
          } else {
            DeltaImgG[index] = BlurImgG[index] / 0.001f;
          }
          if (mBlurImgBufferB[index] > 0.001f) {
            DeltaImgB[index] = BlurImgB[index] / mBlurImgBufferB[index];
          } else {
            DeltaImgB[index] = BlurImgB[index] / 0.001f;
          }
        } else {
          DeltaImgR[index] = BlurImgR[index] - mBlurImgBufferR[index];
          DeltaImgG[index] = BlurImgG[index] - mBlurImgBufferG[index];
          DeltaImgB[index] = BlurImgB[index] - mBlurImgBufferB[index];
        }
      }
    }
    mBlurGenerator.blurRgb(DeltaImgR.data(), DeltaImgG.data(), DeltaImgB.data(),
                           mBlurWeightBuffer.data(), iwidth, iheight,
                           mErrorImgBufferR.data(), mErrorImgBufferG.data(),
                           mErrorImgBufferB.data(), mErrorWeightBuffer.data(),
                           width, height, false);
    ComputeBilaterRegImageGray(DeblurImgR, width, height,
                               BilateralRegImgR.data());
    ComputeBilaterRegImageGray(DeblurImgG, width, height,
                               BilateralRegImgG.data());
    ComputeBilaterRegImageGray(DeblurImgB, width, height,
                               BilateralRegImgB.data());

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
          DeblurImgR[index] *=
              mErrorImgBufferR[index] / (1 + lambda * BilateralRegImgR[index]);
          DeblurImgG[index] *=
              mErrorImgBufferG[index] / (1 + lambda * BilateralRegImgG[index]);
          DeblurImgB[index] *=
              mErrorImgBufferB[index] / (1 + lambda * BilateralRegImgB[index]);
        } else {
          DeblurImgR[index] +=
              mErrorImgBufferR[index] - lambda * BilateralRegImgR[index];
          DeblurImgG[index] +=
              mErrorImgBufferG[index] - lambda * BilateralRegImgG[index];
          DeblurImgB[index] +=
              mErrorImgBufferB[index] - lambda * BilateralRegImgB[index];
        }

        DeblurImgR[index] = std::clamp(DeblurImgR[index], 0.0f, 1.0f);
        DeblurImgG[index] = std::clamp(DeblurImgG[index], 0.0f, 1.0f);
        DeblurImgB[index] = std::clamp(DeblurImgB[index], 0.0f, 1.0f);
      }
    }

    mErrorCalculator.calculateErrorRgb(DeblurImgR, DeblurImgG, DeblurImgB,
                                       width, height);
  }
}

void RLDeblurrerBilateralLaplReg::ProjectiveMotionRLDeblurBilateralLapRegGray(
    float* BlurImg, int iwidth, int iheight, float* DeblurImg, int width,
    int height, int Niter, bool bPoisson, float lambda) {
  int i = 0, t = 1;
  // Parameters are set according to Levin et al Siggraph'07
  float powD = 0.8f, noiseVar = 0.0005f, epilson = t / 255.0f;
  float minWeight =
      exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

  // Bilateral Laplician Regularization
  for (i = 0; i <= t; i++) {
    mBilateralTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    mBilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                          pow(i / 255.0f, powD - 1.0f)) /
                         minWeight;
  }

  ProcessGray(BlurImg, iwidth, iheight, DeblurImg, width, height, Niter,
              bPoisson, lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    mBilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void RLDeblurrerBilateralLaplReg::ProjectiveMotionRLDeblurBilateralLapRegRgb(
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
    mBilateralTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    mBilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                          pow(i / 255.0f, powD - 1.0f)) /
                         minWeight;
  }

  ProcessRgb(BlurImgR, BlurImgG, BlurImgB, iwidth, iheight, DeblurImgR,
             DeblurImgG, DeblurImgB, width, height, Niter, bPoisson, lambda);

  // Restore the original table
  for (i = 0; i < 256; i++) {
    mBilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }
}

void RLDeblurrerBilateralLaplReg::ComputeBilaterRegImageGray(float* Img,
                                                             int width,
                                                             int height,
                                                             float* BRImg) {
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
                  (mBilateralTable[(int)(fabs(Img[iindex] - Img[index]) *
                                         255.0f)] +
                   mBilateralTable[(int)(fabs(Img[iiindex] - Img[index]) *
                                         255.0f)]) *
                  (2 * Img[index] - Img[iindex] - Img[iiindex]);
            }
          }
        }
      }
    }
  }
}

void RLDeblurrerBilateralLaplReg::SetBilateralTable() {
  int i = 0;
  // Parameters are set according to Levin et al Siggraph'07
  // Better result can be obtained by using smaller noiseVar, but for
  // fairness, we use the same setting.
  float noiseVar = 0.005f;

  // Standard Bilateral Weight
  for (i = 0; i < 256; i++) {
    mBilateralTable[i] = exp(-i * i / (noiseVar * 65025.0f));
  }

  // Bilateral Laplician Regularization
  // int t = 1;
  // float powD = 0.8f;
  // float epilson = t / 255.0f;
  // float minWeight =
  //     exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);
  // for (i = 0; i <= t; i++) {
  //   mBilateralTable[i] = 1.0f;
  // }
  // for (i = t + 1; i < 256; i++) {
  //   mBilateralTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
  //                        pow(i / 255.0f, powD - 1.0f)) /
  //                       minWeight;
  // }
}