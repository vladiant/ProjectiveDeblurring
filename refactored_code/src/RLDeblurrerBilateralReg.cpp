#include "RLDeblurrerBilateralReg.hpp"

#include <cmath>
#include <cstring>

BilateralRegularizer::BilateralRegularizer() { SetBilateralTable(); }

void BilateralRegularizer::SetBuffer(int width, int height) {
  const size_t newSize = width * height;

  if (newSize <= mBilateralRegImg.size()) {
    return;
  }

  mBilateralRegImg.resize(newSize);

  mBilateralRegImgR.resize(newSize);
  mBilateralRegImgG.resize(newSize);
  mBilateralRegImgB.resize(newSize);
}

void BilateralRegularizer::ClearBuffer() {
  mBilateralRegImg.clear();

  mBilateralRegImgR.clear();
  mBilateralRegImgG.clear();
  mBilateralRegImgB.clear();
}

void BilateralRegularizer::applyRegularizationGray(float* DeblurImg, int width,
                                                   int height, bool bPoisson,
                                                   float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;

  ComputeBilaterRegImageGray(DeblurImg, width, height, mBilateralRegImg.data());

  for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
    for (x = 0; x < width; x++, index++) {
      if (bPoisson) {
        DeblurImg[index] *= 1.0 / (1.0 + lambda * (mBilateralRegImg[index]));
      } else {
        DeblurImg[index] -= lambda * (mBilateralRegImg[index]);
      }
    }
  }
}

void BilateralRegularizer::applyRegularizationRgb(float* DeblurImgR,
                                                  float* DeblurImgG,
                                                  float* DeblurImgB, int width,
                                                  int height, bool bPoisson,
                                                  float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;

  ComputeBilaterRegImageGray(DeblurImgR, width, height,
                             mBilateralRegImgR.data());
  ComputeBilaterRegImageGray(DeblurImgG, width, height,
                             mBilateralRegImgG.data());
  ComputeBilaterRegImageGray(DeblurImgB, width, height,
                             mBilateralRegImgB.data());

  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (bPoisson) {
        DeblurImgR[index] *= 1.0 / (1.0 + lambda * (mBilateralRegImgR[index]));
        DeblurImgG[index] *= 1.0 / (1.0 + lambda * (mBilateralRegImgG[index]));
        DeblurImgB[index] *= 1.0 / (1.0 + lambda * (mBilateralRegImgB[index]));
      } else {
        DeblurImgR[index] -= lambda * (mBilateralRegImgR[index]);
        DeblurImgG[index] -= lambda * (mBilateralRegImgG[index]);
        DeblurImgB[index] -= lambda * (mBilateralRegImgG[index]);
      }
    }
  }
}

void BilateralRegularizer::ComputeBilaterRegImageGray(float* Img, int width,
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

void BilateralRegularizer::SetBilateralTable() {
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