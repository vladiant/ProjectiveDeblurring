#include "TVRegularizer.hpp"

void TVRegularizer::SetBuffer(int width, int height) {
  const std::size_t newSize = width * height;

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
