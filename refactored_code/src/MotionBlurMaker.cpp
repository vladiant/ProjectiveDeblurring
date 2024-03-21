#include "MotionBlurMaker.hpp"

#include <cmath>
#include <cstdio>

#include "MotionBlurImageGenerator.hpp"

bool setBlur(int aBlurType, MotionBlurImageGenerator& aBlurGenerator) {
  switch (aBlurType) {
    case 0:
      printf("Doll example blurring\n");
      aBlurGenerator.SetGlobalParameters(  //
          10,                              // rotation degree
          1.2f,                            // scale factor
          0.0003f,                         // perspective step x
          0.0006f,                         // perspective step y
          10,                              // translation step x
          20                               // translation step y
      );  // This is parameter setting for doll example
      break;
    case 1:
      printf("Cameraman convergence example blurring\n");
      aBlurGenerator.SetGlobalParameters(  //
          -10,                             // rotation degree
          1.1f,                            // scale factor
          0.0004f,                         //
          0.0002f,                         //
          20,                              //
          -15                              //
      );  // This is parameter setting for Cameraman convergence example
      break;
    case 2:
      printf("Rotational motion example blurring\n");
      //   Testing case for rotational motion
      aBlurGenerator.SetGlobalRotation(10);
      break;
    case 3:
      printf("Zooming motion example blurring\n");
      // Testing case for zooming motion
      aBlurGenerator.SetGlobalScaling(1.2f);
      break;
    case 4:
      printf("Translational motion example blurring\n");
      // Testing case for translational motion
      {
        float deltadx = 0.8f;
        for (int i = 0; i < MotionBlurImageGenerator::NumSamples; i++) {
          float dy =
              5.0f *
              sin((float)(i) / MotionBlurImageGenerator::NumSamples * 2 * M_PI);
          aBlurGenerator.Hmatrix[i].Hmatrix[0][0] = 1;
          aBlurGenerator.Hmatrix[i].Hmatrix[0][1] = 0;
          aBlurGenerator.Hmatrix[i].Hmatrix[0][2] = i * deltadx;
          aBlurGenerator.Hmatrix[i].Hmatrix[1][0] = 0;
          aBlurGenerator.Hmatrix[i].Hmatrix[1][1] = 1;
          aBlurGenerator.Hmatrix[i].Hmatrix[1][2] = dy;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][0] = 0;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][1] = 0;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][2] = 1;
          Homography::MatrixInverse(aBlurGenerator.Hmatrix[i].Hmatrix,
                                    aBlurGenerator.IHmatrix[i].Hmatrix);
        }
      }
      break;
    case 5:
      printf("Projective motion example blurring\n");
      // Testing case for projective motion
      aBlurGenerator.SetGlobalPerspective(0.001f, 0.001f);
      break;
    case 6:
      printf("Case 1-4 example blurring\n");
      // Adjust the parameters for generating test case 1-15
      {
        float deltadx = 0.8f;
        float deltascaling = 0.2f / MotionBlurImageGenerator::NumSamples;
        float deltapx = 0.001f / MotionBlurImageGenerator::NumSamples;
        float deltapy = 0.001f / MotionBlurImageGenerator::NumSamples;
        float deltadegree =
            (10.0f * M_PI / 180.0f) / MotionBlurImageGenerator::NumSamples;
        for (int i = 0; i < MotionBlurImageGenerator::NumSamples; i++) {
          float dy =
              5.0f *
              sin((float)(i) / MotionBlurImageGenerator::NumSamples * 2 * M_PI);
          aBlurGenerator.Hmatrix[i].Hmatrix[0][0] =
              (1 + i * deltascaling) * cos(deltadegree * i);
          aBlurGenerator.Hmatrix[i].Hmatrix[0][1] = sin(deltadegree * i);
          aBlurGenerator.Hmatrix[i].Hmatrix[0][2] = i * deltadx;
          aBlurGenerator.Hmatrix[i].Hmatrix[1][0] = -sin(deltadegree * i);
          aBlurGenerator.Hmatrix[i].Hmatrix[1][1] =
              (1 + i * deltascaling) * cos(deltadegree * i);
          aBlurGenerator.Hmatrix[i].Hmatrix[1][2] = dy;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][0] = i * deltapx;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][1] = i * deltapy;
          aBlurGenerator.Hmatrix[i].Hmatrix[2][2] = 1;
          Homography::MatrixInverse(aBlurGenerator.Hmatrix[i].Hmatrix,
                                    aBlurGenerator.IHmatrix[i].Hmatrix);
        }
      }
      break;

    default:
      printf("Unknown case: %d\n", aBlurType);
      return false;
  }

  return true;
}