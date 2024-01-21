#include "MotionBlurImageGenerator.h"

#include <cmath>
#include <cstring>

#include "BicubicInterpolation.h"
#include "ImResize.h"
#include "bitmap.h"
#include "warping.h"

MotionBlurImageGenerator::MotionBlurImageGenerator() {
  for (int i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = 1;
    Hmatrix[i].Hmatrix[0][1] = 0;
    Hmatrix[i].Hmatrix[0][2] = 0;
    Hmatrix[i].Hmatrix[1][0] = 0;
    Hmatrix[i].Hmatrix[1][1] = 1;
    Hmatrix[i].Hmatrix[1][2] = 0;
    Hmatrix[i].Hmatrix[2][0] = 0;
    Hmatrix[i].Hmatrix[2][1] = 0;
    Hmatrix[i].Hmatrix[2][2] = 1;
    IHmatrix[i].Hmatrix[0][0] = 1;
    IHmatrix[i].Hmatrix[0][1] = 0;
    IHmatrix[i].Hmatrix[0][2] = 0;
    IHmatrix[i].Hmatrix[1][0] = 0;
    IHmatrix[i].Hmatrix[1][1] = 1;
    IHmatrix[i].Hmatrix[1][2] = 0;
    IHmatrix[i].Hmatrix[2][0] = 0;
    IHmatrix[i].Hmatrix[2][1] = 0;
    IHmatrix[i].Hmatrix[2][2] = 1;
  }
}

void MotionBlurImageGenerator::WarpImageGray(float* InputImg,
                                             float* inputWeight, int iwidth,
                                             int iheight, float* OutputImg,
                                             float* outputWeight, int width,
                                             int height, int i) {
  if (i >= 0 && i < NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, Hmatrix[-i]);
  }
}

void MotionBlurImageGenerator::WarpImageRgb(
    float* InputImgR, float* InputImgG, float* InputImgB, float* inputWeight,
    int iwidth, int iheight, float* OutputImgR, float* OutputImgG,
    float* OutputImgB, float* outputWeight, int width, int height, int i) {
  if (i >= 0 && i < NumSamples) {
    warpImageRgb(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
                 OutputImgR, OutputImgG, OutputImgB, outputWeight, width,
                 height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImageRgb(InputImgR, InputImgG, InputImgB, inputWeight, iwidth, iheight,
                 OutputImgR, OutputImgG, OutputImgB, outputWeight, width,
                 height, Hmatrix[-i]);
  }
}

void MotionBlurImageGenerator::GenerateMotionBlurImgGray(
    float* InputImg, float* inputWeight, int iwidth, int iheight,
    float* BlurImg, float* outputWeight, int width, int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  // TODO: Preinitialization, Gray only, check size
  if (mWarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImg, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImageGray(InputImg, inputWeight, iwidth, iheight,
                    mWarpImgBuffer.data(), mWarpWeightBuffer.data(), width,
                    height, i);
    } else {
      WarpImageGray(InputImg, inputWeight, iwidth, iheight,
                    mWarpImgBuffer.data(), mWarpWeightBuffer.data(), width,
                    height, -i);
    }
    for (index = 0; index < totalpixel; index++) {
      BlurImg[index] += mWarpImgBuffer[index] * mWarpWeightBuffer[index];
      outputWeight[index] += mWarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImg[index] /= outputWeight[index];
  }
}

void MotionBlurImageGenerator::GenerateMotionBlurImgRgb(
    float* InputImgR, float* InputImgG, float* InputImgB, float* inputWeight,
    int iwidth, int iheight, float* BlurImgR, float* BlurImgG, float* BlurImgB,
    float* outputWeight, int width, int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

  // TODO: Preinitialization, RGB only, chesk size
  if (mWarpImgBuffer.empty()) {
    SetBuffer(width, height);
  }

  memset(BlurImgR, 0, totalpixel * sizeof(float));
  memset(BlurImgG, 0, totalpixel * sizeof(float));
  memset(BlurImgB, 0, totalpixel * sizeof(float));
  memset(outputWeight, 0, totalpixel * sizeof(float));
  for (i = 0; i < NumSamples; i++) {
    if (bforward) {
      WarpImageRgb(InputImgR, InputImgG, InputImgB, inputWeight, iwidth,
                   iheight, mWarpImgBufferR.data(), mWarpImgBufferG.data(),
                   mWarpImgBufferB.data(), mWarpWeightBuffer.data(), width,
                   height, i);
    } else {
      WarpImageRgb(InputImgR, InputImgG, InputImgB, inputWeight, iwidth,
                   iheight, mWarpImgBufferR.data(), mWarpImgBufferG.data(),
                   mWarpImgBufferB.data(), mWarpWeightBuffer.data(), width,
                   height, -i);
    }

    for (index = 0; index < totalpixel; index++) {
      BlurImgR[index] += mWarpImgBufferR[index] * mWarpWeightBuffer[index];
      BlurImgG[index] += mWarpImgBufferG[index] * mWarpWeightBuffer[index];
      BlurImgB[index] += mWarpImgBufferB[index] * mWarpWeightBuffer[index];
      outputWeight[index] += mWarpWeightBuffer[index];
    }
  }

  for (index = 0; index < totalpixel; index++) {
    BlurImgR[index] /= outputWeight[index];
    BlurImgG[index] /= outputWeight[index];
    BlurImgB[index] /= outputWeight[index];
  }
}

void MotionBlurImageGenerator::SetHomography(Homography H, int i) {
  if (i >= 0 && i < NumSamples) {
    memcpy(Hmatrix[i].Hmatrix[0], H.Hmatrix[0], 3 * sizeof(float));
    memcpy(Hmatrix[i].Hmatrix[1], H.Hmatrix[1], 3 * sizeof(float));
    memcpy(Hmatrix[i].Hmatrix[2], H.Hmatrix[2], 3 * sizeof(float));

    Homography::MatrixInverse(Hmatrix[i].Hmatrix, IHmatrix[i].Hmatrix);
  }
}

void MotionBlurImageGenerator::SetGlobalRotation(float degree) {
  int i;
  float deltadegree = (degree * M_PI / 180.0f) / NumSamples;
  for (i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = cos(deltadegree * i);
    Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
    Hmatrix[i].Hmatrix[0][2] = 0;
    Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
    Hmatrix[i].Hmatrix[1][1] = cos(deltadegree * i);
    Hmatrix[i].Hmatrix[1][2] = 0;
    Hmatrix[i].Hmatrix[2][0] = 0;
    Hmatrix[i].Hmatrix[2][1] = 0;
    Hmatrix[i].Hmatrix[2][2] = 1;
    IHmatrix[i].Hmatrix[0][0] = cos(deltadegree * i);
    IHmatrix[i].Hmatrix[0][1] = -sin(deltadegree * i);
    IHmatrix[i].Hmatrix[0][2] = 0;
    IHmatrix[i].Hmatrix[1][0] = sin(deltadegree * i);
    IHmatrix[i].Hmatrix[1][1] = cos(deltadegree * i);
    IHmatrix[i].Hmatrix[1][2] = 0;
    IHmatrix[i].Hmatrix[2][0] = 0;
    IHmatrix[i].Hmatrix[2][1] = 0;
    IHmatrix[i].Hmatrix[2][2] = 1;
  }
}
void MotionBlurImageGenerator::SetGlobalScaling(float scalefactor) {
  int i;
  float deltascale = (scalefactor - 1.0f) / NumSamples;
  for (i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = 1.0f + i * deltascale;
    Hmatrix[i].Hmatrix[0][1] = 0;
    Hmatrix[i].Hmatrix[0][2] = 0;
    Hmatrix[i].Hmatrix[1][0] = 0;
    Hmatrix[i].Hmatrix[1][1] = 1.0f + i * deltascale;
    Hmatrix[i].Hmatrix[1][2] = 0;
    Hmatrix[i].Hmatrix[2][0] = 0;
    Hmatrix[i].Hmatrix[2][1] = 0;
    Hmatrix[i].Hmatrix[2][2] = 1;
    IHmatrix[i].Hmatrix[0][0] = 1.0f / (1.0f + i * deltascale);
    IHmatrix[i].Hmatrix[0][1] = 0;
    IHmatrix[i].Hmatrix[0][2] = 0;
    IHmatrix[i].Hmatrix[1][0] = 0;
    IHmatrix[i].Hmatrix[1][1] = 1.0f / (1.0f + i * deltascale);
    IHmatrix[i].Hmatrix[1][2] = 0;
    IHmatrix[i].Hmatrix[2][0] = 0;
    IHmatrix[i].Hmatrix[2][1] = 0;
    IHmatrix[i].Hmatrix[2][2] = 1;
  }
}
void MotionBlurImageGenerator::SetGlobalTranslation(float dx, float dy) {
  int i;
  float deltadx = dx / NumSamples;
  float deltady = dy / NumSamples;
  for (i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = 1;
    Hmatrix[i].Hmatrix[0][1] = 0;
    Hmatrix[i].Hmatrix[0][2] = i * deltadx;
    Hmatrix[i].Hmatrix[1][0] = 0;
    Hmatrix[i].Hmatrix[1][1] = 1;
    Hmatrix[i].Hmatrix[1][2] = i * deltady;
    Hmatrix[i].Hmatrix[2][0] = 0;
    Hmatrix[i].Hmatrix[2][1] = 0;
    Hmatrix[i].Hmatrix[2][2] = 1;
    IHmatrix[i].Hmatrix[0][0] = 1;
    IHmatrix[i].Hmatrix[0][1] = 0;
    IHmatrix[i].Hmatrix[0][2] = -i * deltadx;
    IHmatrix[i].Hmatrix[1][0] = 0;
    IHmatrix[i].Hmatrix[1][1] = 1;
    IHmatrix[i].Hmatrix[1][2] = -i * deltady;
    IHmatrix[i].Hmatrix[2][0] = 0;
    IHmatrix[i].Hmatrix[2][1] = 0;
    IHmatrix[i].Hmatrix[2][2] = 1;
  }
}

void MotionBlurImageGenerator::SetGlobalPerspective(float px, float py) {
  int i;
  float deltapx = px / NumSamples;
  float deltapy = py / NumSamples;
  for (i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = 1;
    Hmatrix[i].Hmatrix[0][1] = 0;
    Hmatrix[i].Hmatrix[0][2] = 0;
    Hmatrix[i].Hmatrix[1][0] = 0;
    Hmatrix[i].Hmatrix[1][1] = 1;
    Hmatrix[i].Hmatrix[1][2] = 0;
    Hmatrix[i].Hmatrix[2][0] = i * deltapx;
    Hmatrix[i].Hmatrix[2][1] = i * deltapy;
    Hmatrix[i].Hmatrix[2][2] = 1;
    IHmatrix[i].Hmatrix[0][0] = 1;
    IHmatrix[i].Hmatrix[0][1] = 0;
    IHmatrix[i].Hmatrix[0][2] = 0;
    IHmatrix[i].Hmatrix[1][0] = 0;
    IHmatrix[i].Hmatrix[1][1] = 1;
    IHmatrix[i].Hmatrix[1][2] = 0;
    IHmatrix[i].Hmatrix[2][0] = -i * deltapx;
    IHmatrix[i].Hmatrix[2][1] = -i * deltapy;
    IHmatrix[i].Hmatrix[2][2] = 1;
  }
}

void MotionBlurImageGenerator::SetGlobalParameters(float degree,
                                                   float scalefactor, float px,
                                                   float py, float dx,
                                                   float dy) {
  int i;
  float deltadegree = (degree * M_PI / 180.0f) / NumSamples;
  float deltascale = (scalefactor - 1.0f) / NumSamples;
  float deltapx = px / NumSamples;
  float deltapy = py / NumSamples;
  float deltadx = dx / NumSamples;
  float deltady = dy / NumSamples;
  for (i = 0; i < NumSamples; i++) {
    Hmatrix[i].Hmatrix[0][0] = cos(deltadegree * i) * (1.0f + i * deltascale);
    Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
    Hmatrix[i].Hmatrix[0][2] = i * deltadx;
    Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
    Hmatrix[i].Hmatrix[1][1] = cos(deltadegree * i) * (1.0f + i * deltascale);
    Hmatrix[i].Hmatrix[1][2] = i * deltady;
    Hmatrix[i].Hmatrix[2][0] = i * deltapx;
    Hmatrix[i].Hmatrix[2][1] = i * deltapy;
    Hmatrix[i].Hmatrix[2][2] = 1;

    Homography::MatrixInverse(Hmatrix[i].Hmatrix, IHmatrix[i].Hmatrix);
  }
}

void MotionBlurImageGenerator::SetBuffer(int width, int height) {
  mWarpImgBuffer.resize(width * height);
  mWarpImgBufferR.resize(width * height);
  mWarpImgBufferG.resize(width * height);
  mWarpImgBufferB.resize(width * height);
  mWarpWeightBuffer.resize(width * height);
}

void MotionBlurImageGenerator::ClearBuffer() {
  mWarpImgBuffer.clear();
  mWarpImgBufferR.clear();
  mWarpImgBufferG.clear();
  mWarpImgBufferB.clear();
  mWarpWeightBuffer.clear();
}