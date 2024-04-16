#include "BlurKernelGenerator.hpp"

BlurKernelGenerator::BlurKernelGenerator(int aKernelHalfWidth,
                                         int aKernelHalfHeight, Border aBorder,
                                         float* aBaseImg, int aWidth,
                                         int aHeight)
    : mKernelHalfWidth{aKernelHalfWidth},
      mKernelHalfHeight{aKernelHalfHeight},
      mBorder{aBorder},
      mBaseImg{aBaseImg},
      mWidth{aWidth},
      mHeight{aHeight} {
  // TODO: Add check here
}

float BlurKernelGenerator::getKernelWeightedPoint(float* aKernelImg,
                                                  int xCenter, int yCenter,
                                                  int aXmin, int aXmax,
                                                  int aYmin, int aYmax) const {
  float sum = 0;

  for (int yK = aYmin; yK < aYmax; yK++) {
    float* kernelRow = &aKernelImg[yK * mWidth];
    for (int xK = aXmin; xK < aXmax; xK++) {
      auto kernelWeight = *kernelRow++;
      int xBase = xK + xCenter;
      int yBase = yK + yCenter;

      if (xBase < 0) {
        switch (mBorder) {
          case Border::ISOLATED:
            kernelWeight = 0;
            xBase = 0;
            break;
          case Border::REFLECT:
            xBase = -xBase % mWidth;
            break;
          case Border::REPLICATE:
            xBase = 0;
            break;
          case Border::WRAP:
            const int steps = -xBase / mWidth;
            xBase = (1 + steps) * mWidth - xBase;
            break;
        }
      }

      if (xBase >= mWidth) {
        switch (mBorder) {
          case Border::ISOLATED:
            kernelWeight = 0;
            xBase = 0;
            break;
          case Border::REFLECT:
            xBase = mWidth - (xBase % mWidth);
            break;
          case Border::REPLICATE:
            xBase = mWidth - 1;
            break;
          case Border::WRAP:
            xBase = xBase % mWidth;
            break;
        }
      }

      if (yBase < 0) {
        switch (mBorder) {
          case Border::ISOLATED:
            kernelWeight = 0;
            yBase = 0;
            break;
          case Border::REFLECT:
            yBase = -yBase % mHeight;
            break;
          case Border::REPLICATE:
            yBase = 0;
            break;
          case Border::WRAP:
            const int steps = -yBase / mHeight;
            yBase = (1 + steps) * mHeight - yBase;
            break;
        }
      }

      if (yBase >= mHeight) {
        switch (mBorder) {
          case Border::ISOLATED:
            kernelWeight = 0;
            yBase = 0;
            break;
          case Border::REFLECT:
            yBase = mHeight - (yBase % mHeight);
            break;
          case Border::REPLICATE:
            yBase = mHeight - 1;
            break;
          case Border::WRAP:
            yBase = yBase % mHeight;
            break;
        }
      }

      sum += kernelWeight * mBaseImg[xBase + mWidth * yBase];
    }
  }

  return sum;
}

void BlurKernelGenerator::blurGray(float* InputImg, float* inputWeight,
                                   int iwidth, int iheight, float* BlurImg,
                                   float* outputWeight, int width, int height,
                                   bool bforward) {
  // TODO: Add check here
  // mWidth == width , mHeight = height

  const int transposed = !bforward;
  const int notTransposed = bforward;

  for (int y = 0, index = 0; y < mHeight; y++) {
    for (int x = 0; x < mWidth; x++, index++) {
      const int xCenter = notTransposed * x + transposed * y;
      const int yCenter = notTransposed * y + transposed * x;

      // Kernel application
      float sum = 0;

      // TODO: Consider if (fx >= 0 && fx < iwidth - 1 && fy >= 0 && fy <
      // iheight - 1)

      // Quadrant 1
      sum += getKernelWeightedPoint(InputImg, xCenter, yCenter, 0,
                                    mKernelHalfWidth + 1, 0,
                                    mKernelHalfHeight + 1);

      // Quadrant 2
      sum += getKernelWeightedPoint(InputImg, xCenter, yCenter,
                                    mWidth - mKernelHalfWidth, mWidth, 0,
                                    mKernelHalfHeight + 1);

      // Quadrant 4
      sum += getKernelWeightedPoint(InputImg, xCenter, yCenter, 0,
                                    mKernelHalfWidth + 1,
                                    mHeight - mKernelHalfHeight, mHeight);

      // Quadrant 3
      sum += getKernelWeightedPoint(InputImg, xCenter, yCenter,
                                    mWidth - mKernelHalfWidth, mWidth,
                                    mHeight - mKernelHalfHeight, mHeight);

      BlurImg[index] = sum;

      if (inputWeight) {
        float weightSum = 0;

        // Quadrant 1
        weightSum += getKernelWeightedPoint(InputImg, xCenter, yCenter, 0,
                                            mKernelHalfWidth + 1, 0,
                                            mKernelHalfHeight + 1);

        // Quadrant 2
        weightSum += getKernelWeightedPoint(InputImg, xCenter, yCenter,
                                            mWidth - mKernelHalfWidth, mWidth,
                                            0, mKernelHalfHeight + 1);

        // Quadrant 4
        weightSum += getKernelWeightedPoint(
            InputImg, xCenter, yCenter, 0, mKernelHalfWidth + 1,
            mHeight - mKernelHalfHeight, mHeight);

        // Quadrant 3
        weightSum += getKernelWeightedPoint(
            InputImg, xCenter, yCenter, mWidth - mKernelHalfWidth, mWidth,
            mHeight - mKernelHalfHeight, mHeight);

        outputWeight[index] = sum;
      }
    }
  }
}

void BlurKernelGenerator::blurRgb(float* InputImgR, float* InputImgG,
                                  float* InputImgB, float* inputWeight,
                                  int iwidth, int iheight, float* BlurImgR,
                                  float* BlurImgG, float* BlurImgB,
                                  float* outputWeight, int width, int height,
                                  bool bforward) {}

void BlurKernelGenerator::SetBuffer([[maybe_unused]] int width,
                                    [[maybe_unused]] int height) {}

void BlurKernelGenerator::ClearBuffer() {}