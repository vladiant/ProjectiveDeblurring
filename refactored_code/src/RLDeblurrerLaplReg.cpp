#include "RLDeblurrerLaplReg.hpp"

#include <cmath>

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