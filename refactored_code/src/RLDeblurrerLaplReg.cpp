#include "RLDeblurrerLaplReg.hpp"

#include <cmath>

RLDeblurrerLaplReg::RLDeblurrerLaplReg(IBlurImageGenerator& aBlurGenerator,
                                       IErrorCalculator& aErrorCalculator)
    : mBlurGenerator(aBlurGenerator), mErrorCalculator(aErrorCalculator) {
  SetSpsTable();
}

float RLDeblurrerLaplReg::getSpsWeight(float aValue) const {
  if (!std::isfinite(aValue)) {
    return 0.0f;
  }

  return mSpsTable[static_cast<int>(std::abs(aValue) * 255.0f)];
}

void RLDeblurrerLaplReg::SetSpsTable() {
  int i = 0, t = 1;
  // Parameters, e.g. noiseVar and epilson, are set according to Levin et al
  // Siggraph'07
  float powD = 0.8f, noiseVar = 0.005f, epilson = t / 255.0f;
  float minWeight =
      exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

  for (i = 0; i <= t; i++) {
    mSpsTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    mSpsTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                    pow(i / 255.0f, powD - 1.0f)) /
                   minWeight;

    // Reweighting scheme from Levin et al Siggraph'07,
    // Similar Effect, smaller smoothing weight for large gradient
    // float minEpi = pow(epilson, powD - 2.0f);
    // mSpsTable[i] = pow(i / 255.0f, powD - 2.0f) / minEpi;
  }
}

void RLDeblurrerLaplReg::SetBuffer(int width, int height) {
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

void RLDeblurrerLaplReg::ClearBuffer() {
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

void RLDeblurrerLaplReg::ProjectiveMotionRLDeblurSpsRegGray(
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
              mErrorImgBuffer[index];
        } else {
          DeblurImg[index] = DeblurImg[index] + mErrorImgBuffer[index] -
                             lambda * (Wx * DxxImg[index] + Wy * DyyImg[index]);
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
        if (std::isnan(DeblurImg[index])) DeblurImg[index] = 0;
      }
    }

    mErrorCalculator.calculateErrorGray(DeblurImg, width, height);
  }
}

void RLDeblurrerLaplReg::ProjectiveMotionRLDeblurSpsRegRgb(
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

    ComputeGradientImageGray(DeblurImgR, width, height, DxImgR.data(),
                             DyImgR.data(), true);
    ComputeGradientImageGray(DeblurImgG, width, height, DxImgG.data(),
                             DyImgG.data(), true);
    ComputeGradientImageGray(DeblurImgB, width, height, DxImgB.data(),
                             DyImgB.data(), true);

    ComputeGradientXImageGray(DxImgR.data(), width, height, DxxImgR.data(),
                              false);
    ComputeGradientYImageGray(DyImgR.data(), width, height, DyyImgR.data(),
                              false);
    ComputeGradientXImageGray(DxImgG.data(), width, height, DxxImgG.data(),
                              false);
    ComputeGradientYImageGray(DyImgG.data(), width, height, DyyImgG.data(),
                              false);
    ComputeGradientXImageGray(DxImgB.data(), width, height, DxxImgB.data(),
                              false);
    ComputeGradientYImageGray(DyImgB.data(), width, height, DyyImgB.data(),
                              false);

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
              mErrorImgBufferR[index] /
              (1 + lambda * (WxR * DxxImgR[index] + WyR * DyyImgR[index]));
          DeblurImgG[index] *=
              mErrorImgBufferG[index] /
              (1 + lambda * (WxG * DxxImgG[index] + WyG * DyyImgG[index]));
          DeblurImgB[index] *=
              mErrorImgBufferB[index] /
              (1 + lambda * (WxB * DxxImgB[index] + WyB * DyyImgB[index]));
        } else {
          DeblurImgR[index] +=
              mErrorImgBufferR[index] -
              lambda * (WxR * DxxImgR[index] + WyR * DyyImgR[index]);
          DeblurImgG[index] +=
              mErrorImgBufferG[index] -
              lambda * (WxG * DxxImgG[index] + WyG * DyyImgG[index]);
          DeblurImgB[index] +=
              mErrorImgBufferB[index] -
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

    mErrorCalculator.calculateErrorRgb(DeblurImgR, DeblurImgG, DeblurImgB,
                                       width, height);
  }
}

void RLDeblurrerLaplReg::ComputeGradientImageGray(float* Img, int width,
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

void RLDeblurrerLaplReg::ComputeGradientXImageGray(float* Img, int width,
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

void RLDeblurrerLaplReg::ComputeGradientYImageGray(float* Img, int width,
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

LaplacianRegularizer::LaplacianRegularizer() { SetSpsTable(); }

float LaplacianRegularizer::getSpsWeight(float aValue) const {
  if (!std::isfinite(aValue)) {
    return 0.0f;
  }

  return mSpsTable[static_cast<int>(std::abs(aValue) * 255.0f)];
}

void LaplacianRegularizer::SetSpsTable() {
  int i = 0, t = 1;
  // Parameters, e.g. noiseVar and epilson, are set according to Levin et al
  // Siggraph'07
  float powD = 0.8f, noiseVar = 0.005f, epilson = t / 255.0f;
  float minWeight =
      exp(-pow(epilson, powD) / noiseVar) * pow(epilson, powD - 1.0f);

  for (i = 0; i <= t; i++) {
    mSpsTable[i] = 1.0f;
  }
  for (i = t + 1; i < 256; i++) {
    mSpsTable[i] = (exp(-pow(i / 255.0f, powD) / noiseVar) *
                    pow(i / 255.0f, powD - 1.0f)) /
                   minWeight;

    // Reweighting scheme from Levin et al Siggraph'07,
    // Similar Effect, smaller smoothing weight for large gradient
    // float minEpi = pow(epilson, powD - 2.0f);
    // mSpsTable[i] = pow(i / 255.0f, powD - 2.0f) / minEpi;
  }
}

void LaplacianRegularizer::SetBuffer(int width, int height) {
  const size_t newSize = width * height;

  if (newSize <= mDxImg.size()) {
    return;
  }

  mDxImg.resize(newSize);
  mDyImg.resize(newSize);
  mDxxImg.resize(newSize);
  mDyyImg.resize(newSize);

  mDxImgR.resize(newSize);
  mDyImgR.resize(newSize);
  mDxxImgR.resize(newSize);
  mDyyImgR.resize(newSize);
  mDxImgG.resize(newSize);
  mDyImgG.resize(newSize);
  mDxxImgG.resize(newSize);
  mDyyImgG.resize(newSize);
  mDxImgB.resize(newSize);
  mDyImgB.resize(newSize);
  mDxxImgB.resize(newSize);
  mDyyImgB.resize(newSize);
}

void LaplacianRegularizer::ClearBuffer() {
  mDxImg.clear();
  mDyImg.clear();
  mDxxImg.clear();
  mDyyImg.clear();

  mDxImgR.clear();
  mDyImgR.clear();
  mDxxImgR.clear();
  mDyyImgR.clear();
  mDxImgG.clear();
  mDyImgG.clear();
  mDxxImgG.clear();
  mDyyImgG.clear();
  mDxImgB.clear();
  mDyImgB.clear();
  mDxxImgB.clear();
  mDyyImgB.clear();
}

void LaplacianRegularizer::ComputeGradientImageGray(float* Img, int width,
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

void LaplacianRegularizer::ComputeGradientXImageGray(float* Img, int width,
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

void LaplacianRegularizer::ComputeGradientYImageGray(float* Img, int width,
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

void LaplacianRegularizer::applyRegularizationGray(float* DeblurImg, int width,
                                                   int height, bool bPoisson,
                                                   float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;
  float Wx = NAN, Wy = NAN;

  ComputeGradientImageGray(DeblurImg, width, height, mDxImg.data(),
                           mDyImg.data(), true);
  ComputeGradientXImageGray(mDxImg.data(), width, height, mDxxImg.data(),
                            false);
  ComputeGradientYImageGray(mDyImg.data(), width, height, mDyyImg.data(),
                            false);

  for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
    for (x = 0; x < width; x++, index++) {
      Wx = getSpsWeight(mDxImg[index]);
      Wy = getSpsWeight(mDyImg[index]);
      if (bPoisson) {
        DeblurImg[index] *=
            1.0 / (1.0 + lambda * (Wx * mDxxImg[index] + Wy * mDyyImg[index]));
      } else {
        DeblurImg[index] -=
            lambda * (Wx * mDxxImg[index] + Wy * mDyyImg[index]);
      }
      if (std::isnan(DeblurImg[index])) DeblurImg[index] = 0;
    }
  }
}

void LaplacianRegularizer::applyRegularizationRgb(float* DeblurImgR,
                                                  float* DeblurImgG,
                                                  float* DeblurImgB, int width,
                                                  int height, bool bPoisson,
                                                  float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;
  float WxR = NAN, WyR = NAN, WxG = NAN, WyG = NAN, WxB = NAN, WyB = NAN;

  ComputeGradientImageGray(DeblurImgR, width, height, mDxImgR.data(),
                           mDyImgR.data(), true);
  ComputeGradientImageGray(DeblurImgG, width, height, mDxImgG.data(),
                           mDyImgG.data(), true);
  ComputeGradientImageGray(DeblurImgB, width, height, mDxImgB.data(),
                           mDyImgB.data(), true);

  ComputeGradientXImageGray(mDxImgR.data(), width, height, mDxxImgR.data(),
                            false);
  ComputeGradientYImageGray(mDyImgR.data(), width, height, mDyyImgR.data(),
                            false);
  ComputeGradientXImageGray(mDxImgG.data(), width, height, mDxxImgG.data(),
                            false);
  ComputeGradientYImageGray(mDyImgG.data(), width, height, mDyyImgG.data(),
                            false);
  ComputeGradientXImageGray(mDxImgB.data(), width, height, mDxxImgB.data(),
                            false);
  ComputeGradientYImageGray(mDyImgB.data(), width, height, mDyyImgB.data(),
                            false);

  for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
    for (x = 0; x < width; x++, index++) {
      WxR = getSpsWeight(mDxImgR[index]);
      WyR = getSpsWeight(mDyImgR[index]);
      WxG = getSpsWeight(mDxImgG[index]);
      WyG = getSpsWeight(mDyImgG[index]);
      WxB = getSpsWeight(mDxImgB[index]);
      WyB = getSpsWeight(mDyImgB[index]);

      if (bPoisson) {
        DeblurImgR[index] *=
            1.0 /
            (1.0 + lambda * (WxR * mDxxImgR[index] + WyR * mDyyImgR[index]));
        DeblurImgG[index] *=
            1.0 /
            (1.0 + lambda * (WxG * mDxxImgG[index] + WyG * mDyyImgG[index]));
        DeblurImgB[index] *=
            1.0 /
            (1.0 + lambda * (WxB * mDxxImgB[index] + WyB * mDyyImgB[index]));
      } else {
        DeblurImgR[index] -=
            lambda * (WxR * mDxxImgR[index] + WyR * mDyyImgR[index]);
        DeblurImgG[index] -=
            lambda * (WxG * mDxxImgG[index] + WyG * mDyyImgG[index]);
        DeblurImgB[index] -=
            lambda * (WxB * mDxxImgB[index] + WyB * mDyyImgB[index]);
      }

      if (std::isnan(DeblurImgR[index])) DeblurImgR[index] = 0;
      if (std::isnan(DeblurImgG[index])) DeblurImgG[index] = 0;
      if (std::isnan(DeblurImgB[index])) DeblurImgB[index] = 0;
    }
  }
}