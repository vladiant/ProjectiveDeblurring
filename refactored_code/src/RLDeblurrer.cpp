#include "RLDeblurrer.hpp"

RLDeblurrer::RLDeblurrer(IBlurImageGenerator& aBlurGenerator,
                         IErrorCalculator& aErrorCalculator)
    : mBlurGenerator(aBlurGenerator), mErrorCalculator(aErrorCalculator) {}

void RLDeblurrer::SetBuffer(int width, int height) {
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

void RLDeblurrer::ClearBuffer() {
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

void RLDeblurrer::deblurGray(float* BlurImg, int iwidth, int iheight,
                             float* DeblurImg, int width, int height,
                             const Parameters& aParameters) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);

  ClearBuffer();
  if (width * height >= iwidth * iheight)
    SetBuffer(width, height);
  else
    SetBuffer(iwidth, iheight);

  for (itr = 0; itr < aParameters.Niter; itr++) {
    mBlurGenerator.blurGray(DeblurImg, InputWeight, width, height,
                            mBlurImgBuffer.data(), mBlurWeightBuffer.data(),
                            iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (aParameters.bPoisson) {
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
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (aParameters.bPoisson) {
          DeblurImg[index] *= mErrorImgBuffer[index];
        } else {
          DeblurImg[index] += mErrorImgBuffer[index];
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

    mErrorCalculator.calculateErrorGray(DeblurImg, width, height);
  }
}

void RLDeblurrer::deblurRgb(float* BlurImgR, float* BlurImgG, float* BlurImgB,
                            int iwidth, int iheight, float* DeblurImgR,
                            float* DeblurImgG, float* DeblurImgB, int width,
                            int height, const Parameters& aParameters) {
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

  for (itr = 0; itr < aParameters.Niter; itr++) {
    mBlurGenerator.blurRgb(DeblurImgR, DeblurImgG, DeblurImgB, InputWeight,
                           width, height, mBlurImgBufferR.data(),
                           mBlurImgBufferG.data(), mBlurImgBufferB.data(),
                           mBlurWeightBuffer.data(), iwidth, iheight, true);
    for (y = 0, index = 0; y < iheight; y++) {
      for (x = 0; x < iwidth; x++, index++) {
        if (aParameters.bPoisson) {
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
    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (aParameters.bPoisson) {
          DeblurImgR[index] *= mErrorImgBufferR[index];
          DeblurImgG[index] *= mErrorImgBufferG[index];
          DeblurImgB[index] *= mErrorImgBufferB[index];
        } else {
          DeblurImgR[index] += mErrorImgBufferR[index];
          DeblurImgG[index] += mErrorImgBufferG[index];
          DeblurImgB[index] += mErrorImgBufferB[index];
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
