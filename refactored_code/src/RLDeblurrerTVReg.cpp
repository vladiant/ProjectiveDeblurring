#include "RLDeblurrerTVReg.hpp"

RLDeblurrerTVReg::RLDeblurrerTVReg(IBlurImageGenerator& aBlurGenerator,
                                   IErrorCalculator& aErrorCalculator)
    : mBlurGenerator(aBlurGenerator), mErrorCalculator(aErrorCalculator) {}

void RLDeblurrerTVReg::SetBuffer(int width, int height) {
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

void RLDeblurrerTVReg::ClearBuffer() {
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

void RLDeblurrerTVReg::ProjectiveMotionRLDeblurTVRegGray(
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
          DeblurImg[index] *= mErrorImgBuffer[index] /
                              (1 + lambda * (DxxImg[index] + DyyImg[index]));
        } else {
          DeblurImg[index] +=
              mErrorImgBuffer[index] - lambda * (DxxImg[index] + DyyImg[index]);
        }
        DeblurImg[index] = std::clamp(DeblurImg[index], 0.0f, 1.0f);
      }
    }

    mErrorCalculator.calculateErrorGray(DeblurImg, width, height);
  }
}

void RLDeblurrerTVReg::ProjectiveMotionRLDeblurTVRegRgb(
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
        if (bPoisson) {
          DeblurImgR[index] *= mErrorImgBufferR[index] /
                               (1 + lambda * (DxxImgR[index] + DyyImgR[index]));
          DeblurImgG[index] *= mErrorImgBufferG[index] /
                               (1 + lambda * (DxxImgG[index] + DyyImgG[index]));
          DeblurImgB[index] *= mErrorImgBufferB[index] /
                               (1 + lambda * (DxxImgB[index] + DyyImgB[index]));
        } else {
          DeblurImgR[index] += mErrorImgBufferR[index] -
                               lambda * (DxxImgR[index] + DyyImgR[index]);
          DeblurImgG[index] += mErrorImgBufferG[index] -
                               lambda * (DxxImgG[index] + DyyImgG[index]);
          DeblurImgB[index] += mErrorImgBufferB[index] -
                               lambda * (DxxImgB[index] + DyyImgB[index]);
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

void RLDeblurrerTVReg::ComputeGradientImageGray(float* Img, int width,
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

void RLDeblurrerTVReg::ComputeGradientXImageGray(float* Img, int width,
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

void RLDeblurrerTVReg::ComputeGradientYImageGray(float* Img, int width,
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