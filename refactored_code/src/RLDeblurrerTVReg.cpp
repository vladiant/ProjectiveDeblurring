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
    int height, int Niter, bool bPoisson, float lambda,
    IRegularizer& regularizer) {
  int x = 0, y = 0, index = 0, itr = 0;
  float* InputWeight = nullptr;

  std::vector<float> DeltaImg(iwidth * iheight);

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

    regularizer.applyRegularizationGray(DeblurImg, width, height, bPoisson,
                                        lambda);

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

    mErrorCalculator.calculateErrorGray(DeblurImg, width, height);
  }
}

void RLDeblurrerTVReg::ProjectiveMotionRLDeblurTVRegRgb(
    float* BlurImgR, float* BlurImgG, float* BlurImgB, int iwidth, int iheight,
    float* DeblurImgR, float* DeblurImgG, float* DeblurImgB, int width,
    int height, int Niter, bool bPoisson, float lambda,
    IRegularizer& regularizer) {
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

    regularizer.applyRegularizationRgb(DeblurImgR, DeblurImgG, DeblurImgB,
                                       width, height, bPoisson, lambda);

    for (y = 0, index = 0; y < height; y++) {
      for (x = 0; x < width; x++, index++) {
        if (bPoisson) {
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

void TVRegularizer::SetBuffer(int width, int height) {
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

void TVRegularizer::ClearBuffer() {
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

void TVRegularizer::ComputeGradientImageGray(float* Img, int width, int height,
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

void TVRegularizer::ComputeGradientXImageGray(float* Img, int width, int height,
                                              float* DxImg, bool bflag) {
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

void TVRegularizer::ComputeGradientYImageGray(float* Img, int width, int height,
                                              float* DyImg, bool bflag) {
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

void TVRegularizer::applyRegularizationGray(float* DeblurImg, int width,
                                            int height, bool bPoisson,
                                            float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;

  ComputeGradientImageGray(DeblurImg, width, height, mDxImg.data(),
                           mDyImg.data(), true);
  for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
    for (x = 0; x < width; x++, index++) {
      if (mDxImg[index] > 0) mDxImg[index] = 1.0f / 255.0f;
      if (mDxImg[index] < 0) mDxImg[index] = -1.0f / 255.0f;
      if (mDyImg[index] > 0) mDyImg[index] = 1.0f / 255.0f;
      if (mDyImg[index] < 0) mDyImg[index] = -1.0f / 255.0f;
    }
  }
  ComputeGradientXImageGray(mDxImg.data(), width, height, mDxxImg.data(),
                            false);
  ComputeGradientYImageGray(mDyImg.data(), width, height, mDyyImg.data(),
                            false);

  for (y = 0, index = 0; y < height; y++) {  // Normalize the gradient
    for (x = 0; x < width; x++, index++) {
      if (bPoisson) {
        DeblurImg[index] *=
            1.0 / (1.0 + lambda * (mDxxImg[index] + mDyyImg[index]));
      } else {
        DeblurImg[index] -= lambda * (mDxxImg[index] + mDyyImg[index]);
      }
    }
  }
}

void TVRegularizer::applyRegularizationRgb(float* DeblurImgR, float* DeblurImgG,
                                           float* DeblurImgB, int width,
                                           int height, bool bPoisson,
                                           float lambda) {
  SetBuffer(width, height);

  int x = 0, y = 0, index = 0;

  ComputeGradientImageGray(DeblurImgR, width, height, mDxImgR.data(),
                           mDyImgR.data(), true);
  ComputeGradientImageGray(DeblurImgG, width, height, mDxImgG.data(),
                           mDyImgG.data(), true);
  ComputeGradientImageGray(DeblurImgB, width, height, mDxImgB.data(),
                           mDyImgB.data(), true);
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (mDxImgR[index] > 0) mDxImgR[index] = 1.0f / 255.0f;
      if (mDxImgR[index] < 0) mDxImgR[index] = -1.0f / 255.0f;
      if (mDyImgR[index] > 0) mDyImgR[index] = 1.0f / 255.0f;
      if (mDyImgR[index] < 0) mDyImgR[index] = -1.0f / 255.0f;
      if (mDxImgG[index] > 0) mDxImgG[index] = 1.0f / 255.0f;
      if (mDxImgG[index] < 0) mDxImgG[index] = -1.0f / 255.0f;
      if (mDyImgG[index] > 0) mDyImgG[index] = 1.0f / 255.0f;
      if (mDyImgG[index] < 0) mDyImgG[index] = -1.0f / 255.0f;
      if (mDxImgB[index] > 0) mDxImgB[index] = 1.0f / 255.0f;
      if (mDxImgB[index] < 0) mDxImgB[index] = -1.0f / 255.0f;
      if (mDyImgB[index] > 0) mDyImgB[index] = 1.0f / 255.0f;
      if (mDyImgB[index] < 0) mDyImgB[index] = -1.0f / 255.0f;
    }
  }
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
      if (bPoisson) {
        DeblurImgR[index] *=
            1.0 / (1.0 + lambda * (mDxxImgR[index] + mDyyImgR[index]));
        DeblurImgG[index] *=
            1.0 / (1.0 + lambda * (mDxxImgG[index] + mDyyImgG[index]));
        DeblurImgB[index] *=
            1.0 / (1.0 + lambda * (mDxxImgB[index] + mDyyImgB[index]));
      } else {
        DeblurImgR[index] -= lambda * (mDxxImgR[index] + mDyyImgR[index]);
        DeblurImgG[index] -= lambda * (mDxxImgG[index] + mDyyImgG[index]);
        DeblurImgB[index] -= lambda * (mDxxImgB[index] + mDyyImgB[index]);
      }
    }
  }
}
