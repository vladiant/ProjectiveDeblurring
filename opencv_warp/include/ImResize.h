#pragma once

void ImResize(float* Img, int width, int height, float* Rimg, int Rwidth,
              int Rheight);
void ImChoppingGray(float* Img, int width, int height, float* Rimg, int Rwidth,
                    int Rheight);
void ImChoppingGray(float* Img, int width, int height, float* Rimg, int Rwidth,
                    int Rheight, int CenterX, int CenterY);
