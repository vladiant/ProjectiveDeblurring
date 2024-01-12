#include "warping.h"

#include <future>
#include <opencv2/imgproc.hpp>

#include "BicubicInterpolation.h"

constexpr static std::launch kExecutionPolicy = std::launch::async;

void warpImage(float* InputImg, float* inputWeight, int iwidth, int iheight,
               float* OutputImg, float* outputWeight, int width, int height,
               float hmatrix[3][3]) {
  const float woffset = width * 0.5f;
  const float hoffset = height * 0.5f;
  const float iwoffset = iwidth * 0.5f;
  const float ihoffset = iheight * 0.5f;

  const cv::Mat hMat(3, 3, CV_32FC1, hmatrix);

  float transf[9] = {1.0, 0.0, -woffset, 0.0, 1.0, -hoffset, 0.0, 0.0, 1.0};
  const cv::Mat coordTransf(3, 3, CV_32FC1, transf);
  float backTransf[9] = {1.0, 0.0, iwoffset, 0.0, 1.0, ihoffset, 0.0, 0.0, 1.0};
  const cv::Mat coordBackTransf(3, 3, CV_32FC1, backTransf);
  const cv::Mat perspMat = coordBackTransf * hMat * coordTransf;

  const auto warpImg = [&]() {
    const cv::Mat inImg(height, width, CV_32FC1, InputImg);
    const cv::Mat outImg(height, width, CV_32FC1, OutputImg);
    cv::warpPerspective(inImg, outImg, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  };
  auto warpImgTask = std::async(kExecutionPolicy, warpImg);

  const cv::Mat outWeight(height, width, CV_32FC1, outputWeight);
  const cv::Mat inWeight(height, width, CV_32FC1, inputWeight);
  cv::warpPerspective(inWeight + 0.01, outWeight, perspMat,
                      cv::Size(iwidth, iheight),
                      cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                      cv::BORDER_CONSTANT, cv::Scalar(0.01));

  warpImgTask.get();
}

void warpImage(float* InputImgR, float* InputImgG, float* InputImgB,
               float* inputWeight, int iwidth, int iheight, float* OutputImgR,
               float* OutputImgG, float* OutputImgB, float* outputWeight,
               int width, int height, float hmatrix[3][3]) {
  const float woffset = width * 0.5f;
  const float hoffset = height * 0.5f;
  const float iwoffset = iwidth * 0.5f;
  const float ihoffset = iheight * 0.5f;

  const cv::Mat inImgR(height, width, CV_32FC1, InputImgR);
  const cv::Mat inImgG(height, width, CV_32FC1, InputImgG);
  const cv::Mat inImgB(height, width, CV_32FC1, InputImgB);
  const cv::Mat outImgR(height, width, CV_32FC1, OutputImgR);
  const cv::Mat outImgG(height, width, CV_32FC1, OutputImgG);
  const cv::Mat outImgB(height, width, CV_32FC1, OutputImgB);
  const cv::Mat hMat(3, 3, CV_32FC1, hmatrix);

  float transf[9] = {1.0, 0.0, -woffset, 0.0, 1.0, -hoffset, 0.0, 0.0, 1.0};
  const cv::Mat coordTransf(3, 3, CV_32FC1, transf);
  float backTransf[9] = {1.0, 0.0, iwoffset, 0.0, 1.0, ihoffset, 0.0, 0.0, 1.0};
  const cv::Mat coordBackTransf(3, 3, CV_32FC1, backTransf);
  const cv::Mat perspMat = coordBackTransf * hMat * coordTransf;

  const auto warpImgR = [&]() {
    cv::warpPerspective(inImgR, outImgR, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  };

  const auto warpImgG = [&]() {
    cv::warpPerspective(inImgG, outImgG, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  };

  const auto warpImgB = [&]() {
    cv::warpPerspective(inImgB, outImgB, perspMat, cv::Size(iwidth, iheight),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  };

  auto warpImgTaskR = std::async(kExecutionPolicy, warpImgR);
  auto warpImgTaskG = std::async(kExecutionPolicy, warpImgG);
  auto warpImgTaskB = std::async(kExecutionPolicy, warpImgB);

  const cv::Mat outWeight(height, width, CV_32FC1, outputWeight);
  const cv::Mat inWeight(height, width, CV_32FC1, inputWeight);
  cv::warpPerspective(inWeight + 0.01, outWeight, perspMat,
                      cv::Size(iwidth, iheight),
                      cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                      cv::BORDER_CONSTANT, cv::Scalar(0.01));

  warpImgTaskR.get();
  warpImgTaskG.get();
  warpImgTaskB.get();
}