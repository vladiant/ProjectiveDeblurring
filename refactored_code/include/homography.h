#pragma once

#include <array>
#include <vector>

class Homography {
 public:
  // From B to A
  void ComputeHomography(const double (&correspondants)[4][4]);
  void ComputeHomography(const double (&correspondantsA)[4][2],
                         const double (&correspondantsB)[4][2]);
  void ComputeHomography(const double (&correspondantsA)[4][2],
                         const double (&correspondantsB)[4][2],
                         double** featurevector, double* w, double** v,
                         double* rv1);
  void ComputeHomography(
      const std::vector<std::array<double, 4>>& correspondants);
  void ComputeHomography(
      const std::vector<std::array<double, 2>>& correspondantsA,
      const std::vector<std::array<double, 2>>& correspondantsB);

  void ComputeAffineHomography(const double (&correspondants)[4][4]);
  void ComputeAffineHomography(const double (&correspondantsA)[4][2],
                               const double (&correspondantsB)[4][2]);
  void ComputeAffineHomography(
      const std::vector<std::array<double, 4>>& correspondants);
  void ComputeAffineHomography(
      const std::vector<std::array<double, 2>>& correspondantsA,
      const std::vector<std::array<double, 2>>& correspondantsB);

  void ComputeRTHomography(const double (&correspondants)[4][4]);
  void ComputeRTHomography(const double (&correspondantsA)[4][2],
                           const double (&correspondantsB)[4][2]);
  void ComputeRTHomography(
      const std::vector<std::array<double, 4>>& correspondants);
  void ComputeRTHomography(
      const std::vector<std::array<double, 2>>& correspondantsA,
      const std::vector<std::array<double, 2>>& correspondantsB);

  static void MatrixInverse(const float (&A)[3][3], float (&I)[3][3]) {
    const double detA =
        A[0][0] * A[1][1] * A[2][2] + A[1][0] * A[2][1] * A[0][2] +
        A[0][1] * A[1][2] * A[2][0] - A[0][0] * A[1][2] * A[2][1] -
        A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[2][0] * A[1][1];

    I[0][0] = (float)((A[1][1] * A[2][2] - A[1][2] * A[2][1]) / detA);
    I[0][1] = (float)((A[0][2] * A[2][1] - A[0][1] * A[2][2]) / detA);
    I[0][2] = (float)((A[0][1] * A[1][2] - A[0][2] * A[1][1]) / detA);
    I[1][0] = (float)((A[1][2] * A[2][0] - A[1][0] * A[2][2]) / detA);
    I[1][1] = (float)((A[0][0] * A[2][2] - A[0][2] * A[2][0]) / detA);
    I[1][2] = (float)((A[0][2] * A[1][0] - A[0][0] * A[1][2]) / detA);
    I[2][0] = (float)((A[1][0] * A[2][1] - A[1][1] * A[2][0]) / detA);
    I[2][1] = (float)((A[0][1] * A[2][0] - A[0][0] * A[2][1]) / detA);
    I[2][2] = (float)((A[0][0] * A[1][1] - A[0][1] * A[1][0]) / detA);
  }

  void Transform(float& x, float& y) const {
    const float z = Hmatrix[2][0] * x + Hmatrix[2][1] * y + Hmatrix[2][2];
    const float fx =
        (Hmatrix[0][0] * x + Hmatrix[0][1] * y + Hmatrix[0][2]) / z;
    const float fy =
        (Hmatrix[1][0] * x + Hmatrix[1][1] * y + Hmatrix[1][2]) / z;
    x = fx;
    y = fy;
  }

  /////////
  //  A[0]         H[0][0] H[0][1] H[0][2]      B[0]
  //[ A[1] ] =  [  H[1][0] H[1][1] H[1][2]  ] [ B[1] ]
  //  A[2]         H[2][0] H[2][1] H[2][2]      B[2]
  /////////

  float Hmatrix[3][3];
};
