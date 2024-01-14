#include "ImResize.h"

#include <cmath>

#include "BicubicInterpolation.h"

void ImResize(float* Img, int width, int height, float* Rimg, int Rwidth,
              int Rheight) {
  int x = 0, y = 0, index = 0;
  float fx = NAN, fy = NAN;
  float widthscale = (width - 1.0f) / (Rwidth - 1.0f),
        heightscale = (height - 1.0f) / (Rheight - 1.0f);
  for (y = 0, index = 0; y < Rheight; y++) {
    for (x = 0; x < Rwidth; x++, index++) {
      fx = (x)*widthscale;
      fy = (y)*heightscale;
      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx > width - 1.0001f) fx = width - 1.0001f;
      if (fy > height - 1.0001f) fy = height - 1.0001f;

      Rimg[index] = ReturnInterpolatedValue(fx, fy, Img, width, height);
    }
  }
}

void ImChoppingGray(float* Img, int width, int height, float* Rimg, int Rwidth,
                    int Rheight) {
  int x = 0, y = 0, index = 0, lindex = 0;
  int shiftx = (Rwidth - width) / 2, shifty = (Rheight - height) / 2;

  for (y = 0, index = 0; y < Rheight; y++) {
    for (x = 0; x < Rwidth; x++, index++) {
      lindex = 0;
      if (y >= shifty && y < height + shifty) {
        lindex += (y - shifty) * width;
      } else if (y >= height + shifty) {
        lindex += (height - 1) * width;
      }
      if (x >= shiftx && x < width + shiftx) {
        lindex += x - shiftx;
      } else if (x >= width + shiftx) {
        lindex += width - 1;
      }

      Rimg[index] = Img[lindex];
    }
  }
}

void ImChoppingGray(float* Img, int width, int height, float* Rimg, int Rwidth,
                    int Rheight, int CenterX, int CenterY) {
  int x = 0, y = 0, index = 0, lindex = 0;
  int shiftx = (Rwidth - width) / 2 + width / 2 - CenterX,
      shifty = (Rheight - height) / 2 + height / 2 - CenterY;

  for (y = 0, index = 0; y < Rheight; y++) {
    for (x = 0; x < Rwidth; x++, index++) {
      lindex = 0;
      if (y >= shifty && y < height + shifty) {
        lindex += (y - shifty) * width;
      } else if (y >= height + shifty) {
        lindex += (height - 1) * width;
      }
      if (x >= shiftx && x < width + shiftx) {
        lindex += x - shiftx;
      } else if (x >= width + shiftx) {
        lindex += width - 1;
      }

      Rimg[index] = Img[lindex];
    }
  }
}