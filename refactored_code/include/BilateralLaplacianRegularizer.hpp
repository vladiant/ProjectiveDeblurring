#pragma once

#include <vector>

#include "IRegularizer.hpp"

class BilateralLaplacianRegularizer : public IRegularizer {
 public:
  BilateralLaplacianRegularizer();
  ~BilateralLaplacianRegularizer() override { ClearBuffer(); }

  ////////////////////////////////////
  // These functions are used to set Buffer for caching
  ////////////////////////////////////
  void SetBuffer(int width, int height);
  void ClearBuffer();

  ////////////////////////////////////
  // These functions are used to apply regularization
  ////////////////////////////////////
  void applyRegularizationGray(float* DeblurImg, int width, int height,
                               bool bPoisson, float lambda) override;

  void applyRegularizationRgb(float* DeblurImgR, float* DeblurImgG,
                              float* DeblurImgB, int width, int height,
                              bool bPoisson, float lambda) override;

 private:
  void SetBilateralTable();

  // These are buffer and lookup table variables
  float mBilateralTable[256]{};

  ////////////////////////////////////
  // These functions are used to compute derivatives for regularization
  ////////////////////////////////////
  void ComputeBilaterRegImageGray(float* Img, int width, int height,
                                  float* BRImg);

  std::vector<float> mBilateralRegImg;
  std::vector<float> mBilateralRegImgR;
  std::vector<float> mBilateralRegImgG;
  std::vector<float> mBilateralRegImgB;
};
