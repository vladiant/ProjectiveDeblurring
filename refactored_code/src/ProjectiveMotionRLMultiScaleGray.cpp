#include "ProjectiveMotionRLMultiScaleGray.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "BicubicInterpolation.h"
#include "warping.h"

ProjectiveMotionRLMultiScaleGray::ProjectiveMotionRLMultiScaleGray() {
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

ProjectiveMotionRLMultiScaleGray::~ProjectiveMotionRLMultiScaleGray() {
  ClearBuffer();
}

void ProjectiveMotionRLMultiScaleGray::GenerateMotionBlurImgGray(
    float* InputImg, float* inputWeight, int iwidth, int iheight,
    float* BlurImg, float* outputWeight, int width, int height, bool bforward) {
  int i = 0, index = 0, totalpixel = width * height;

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

void ProjectiveMotionRLMultiScaleGray::ProjectiveMotionRLDeblurMultiScaleGray(
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
      // printf("%d\n", itr);
      GenerateMotionBlurImgGray(DeblurImg, InputWeight, width, height,
                                mBlurImgBuffer.data(), mBlurWeightBuffer.data(),
                                bwidth, bheight, true);
      for (y = 0, index = 0; y < bheight; y++) {
        for (x = 0; x < bwidth; x++, index++) {
          float bvalue = ReturnInterpolatedValueFast(x * bfactw, y * bfacth,
                                                     BlurImg, iwidth, iheight);
          if (bPoisson) {
            if (mBlurImgBuffer[index] > 0.0001f) {
              DeltaImg[index] = bvalue / mBlurImgBuffer[index];
            } else {
              DeltaImg[index] = bvalue / 0.0001f;
            }
          } else {
            DeltaImg[index] = bvalue - mBlurImgBuffer[index];
          }
        }
      }
      GenerateMotionBlurImgGray(DeltaImg.data(), mBlurWeightBuffer.data(),
                                bwidth, bheight, mErrorImgBuffer.data(),
                                mErrorWeightBuffer.data(), width, height,
                                false);
      for (y = 0, index = 0; y < height; y++) {
        for (x = 0; x < width; x++, index++) {
          if (bPoisson) {
            DeblurImg[index] *= mErrorImgBuffer[index];
          } else {
            DeblurImg[index] += mErrorImgBuffer[index];
          }
          DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
        }
      }
    }
  }
}

void ProjectiveMotionRLMultiScaleGray::WarpImageGray(
    float* InputImg, float* inputWeight, int iwidth, int iheight,
    float* OutputImg, float* outputWeight, int width, int height, int i) {
  if (i >= 0 && i < NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, IHmatrix[i]);
  } else if (i < 0 && i > -NumSamples) {
    warpImageGray(InputImg, inputWeight, iwidth, iheight, OutputImg,
                  outputWeight, width, height, Hmatrix[-i]);
  }
}