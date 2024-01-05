#include "homography.h"

#include <cmath>

#include "svdcmp.h"

void Homography::ComputeHomography(const double (&correspondants)[4][4]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 9>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2 * sx1;
    featurevector[2 * k][7] = -sx2 * sy1;
    featurevector[2 * k][8] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2 * sx1;
    featurevector[2 * k + 1][7] = -sy2 * sy1;
    featurevector[2 * k + 1][8] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[9];
  double v[9][9];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 9; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex]);
  Hmatrix[2][0] = (float)(v[6][minwindex]);
  Hmatrix[2][1] = (float)(v[7][minwindex]);
  Hmatrix[2][2] = (float)(v[8][minwindex]);
}

void Homography::ComputeHomography(double const (&correspondantsA)[4][2],
                                   double const (&correspondantsB)[4][2]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 9>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2 * sx1;
    featurevector[2 * k][7] = -sx2 * sy1;
    featurevector[2 * k][8] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2 * sx1;
    featurevector[2 * k + 1][7] = -sy2 * sy1;
    featurevector[2 * k + 1][8] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[9];
  double v[9][9];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 9; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex]);
  Hmatrix[2][0] = (float)(v[6][minwindex]);
  Hmatrix[2][1] = (float)(v[7][minwindex]);
  Hmatrix[2][2] = (float)(v[8][minwindex]);
}

void Homography::ComputeHomography(
    double const (&correspondantsA)[4][2],
    double const (&correspondantsB)[4][2],
    std::vector<std::array<double, 9>>& featurevector, double (&w)[9],
    double (&v)[9][9], double (&rv1)[9]) {
  if (featurevector.size() != 8) {
    return;
  }

  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  for (k = 0; k < 4; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2 * sx1;
    featurevector[2 * k][7] = -sx2 * sy1;
    featurevector[2 * k][8] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2 * sx1;
    featurevector[2 * k + 1][7] = -sy2 * sy1;
    featurevector[2 * k + 1][8] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v, rv1);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 9; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex]);
  Hmatrix[2][0] = (float)(v[6][minwindex]);
  Hmatrix[2][1] = (float)(v[7][minwindex]);
  Hmatrix[2][2] = (float)(v[8][minwindex]);
}

void Homography::ComputeHomography(
    const std::vector<std::array<double, 4>>& correspondants) {
  const int ncor = correspondants.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 9>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2 * sx1;
    featurevector[2 * k][7] = -sx2 * sy1;
    featurevector[2 * k][8] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2 * sx1;
    featurevector[2 * k + 1][7] = -sy2 * sy1;
    featurevector[2 * k + 1][8] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[9];
  double v[9][9];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 9; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex]);
  Hmatrix[2][0] = (float)(v[6][minwindex]);
  Hmatrix[2][1] = (float)(v[7][minwindex]);
  Hmatrix[2][2] = (float)(v[8][minwindex]);
}

void Homography::ComputeHomography(
    const std::vector<std::array<double, 2>>& correspondantsA,
    const std::vector<std::array<double, 2>>& correspondantsB) {
  if (correspondantsA.size() != correspondantsB.size()) {
    return;
  }

  const int ncor = correspondantsA.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 9>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2 * sx1;
    featurevector[2 * k][7] = -sx2 * sy1;
    featurevector[2 * k][8] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2 * sx1;
    featurevector[2 * k + 1][7] = -sy2 * sy1;
    featurevector[2 * k + 1][8] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[9];
  double v[9][9];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 9; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex]);
  Hmatrix[2][0] = (float)(v[6][minwindex]);
  Hmatrix[2][1] = (float)(v[7][minwindex]);
  Hmatrix[2][2] = (float)(v[8][minwindex]);
}

void Homography::ComputeAffineHomography(const double (&correspondants)[4][4]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 7>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[7];
  double v[7][7];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 7; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[6][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[6][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[6][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex] / v[6][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex] / v[6][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex] / v[6][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeAffineHomography(
    double const (&correspondantsA)[4][2],
    double const (&correspondantsB)[4][2]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 7>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[7];
  double v[7][7];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 7; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[6][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[6][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[6][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex] / v[6][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex] / v[6][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex] / v[6][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeAffineHomography(
    const std::vector<std::array<double, 4>>& correspondants) {
  const int ncor = correspondants.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 7>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[7];
  double v[7][7];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 7; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[6][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[6][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[6][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex] / v[6][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex] / v[6][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex] / v[6][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeAffineHomography(
    const std::vector<std::array<double, 2>>& correspondantsA,
    const std::vector<std::array<double, 2>>& correspondantsB) {
  if (correspondantsA.size() != correspondantsB.size()) {
    return;
  }

  const int ncor = correspondantsA.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 7>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = 0;
    featurevector[2 * k][5] = 0;
    featurevector[2 * k][6] = -sx2;

    featurevector[2 * k + 1][0] = 0;
    featurevector[2 * k + 1][1] = 0;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = sx1;
    featurevector[2 * k + 1][4] = sy1;
    featurevector[2 * k + 1][5] = 1;
    featurevector[2 * k + 1][6] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[7];
  double v[7][7];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 7; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[6][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[6][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[6][minwindex]);
  Hmatrix[1][0] = (float)(v[3][minwindex] / v[6][minwindex]);
  Hmatrix[1][1] = (float)(v[4][minwindex] / v[6][minwindex]);
  Hmatrix[1][2] = (float)(v[5][minwindex] / v[6][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeRTHomography(const double (&correspondants)[4][4]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 5>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = -sx2;

    featurevector[2 * k + 1][0] = sy1;
    featurevector[2 * k + 1][1] = sx1;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = 1;
    featurevector[2 * k + 1][4] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[5];
  double v[5][5];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 5; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[4][minwindex]);
  Hmatrix[1][0] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[1][1] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[3][minwindex] / v[4][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeRTHomography(double const (&correspondantsA)[4][2],
                                     double const (&correspondantsB)[4][2]) {
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 5>> featurevector(8);
  for (k = 0; k < 4; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = -sx2;

    featurevector[2 * k + 1][0] = sy1;
    featurevector[2 * k + 1][1] = sx1;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = 1;
    featurevector[2 * k + 1][4] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[5];
  double v[5][5];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 5; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[4][minwindex]);
  Hmatrix[1][0] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[1][1] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[3][minwindex] / v[4][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeRTHomography(
    const std::vector<std::array<double, 4>>& correspondants) {
  const int ncor = correspondants.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 5>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondants[k][0];
    sy1 = correspondants[k][1];
    sx2 = correspondants[k][2];
    sy2 = correspondants[k][3];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = -sx2;

    featurevector[2 * k + 1][0] = sy1;
    featurevector[2 * k + 1][1] = sx1;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = 1;
    featurevector[2 * k + 1][4] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[5];
  double v[5][5];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 5; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[4][minwindex]);
  Hmatrix[1][0] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[1][1] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[3][minwindex] / v[4][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}

void Homography::ComputeRTHomography(
    const std::vector<std::array<double, 2>>& correspondantsA,
    const std::vector<std::array<double, 2>>& correspondantsB) {
  if (correspondantsA.size() != correspondantsB.size()) {
    return;
  }

  const int ncor = correspondantsA.size();
  int k = 0;
  double sx1 = NAN, sy1 = NAN, sx2 = NAN, sy2 = NAN;

  /////////////////////////////////////////////////////////////
  // Map correspondants to feature
  ////////////////////////////////////////////////////////////
  std::vector<std::array<double, 5>> featurevector(2 * ncor);
  for (k = 0; k < ncor; k++) {
    sx1 = correspondantsA[k][0];
    sy1 = correspondantsA[k][1];
    sx2 = correspondantsB[k][0];
    sy2 = correspondantsB[k][1];
    featurevector[2 * k][0] = sx1;
    featurevector[2 * k][1] = sy1;
    featurevector[2 * k][2] = 1;
    featurevector[2 * k][3] = 0;
    featurevector[2 * k][4] = -sx2;

    featurevector[2 * k + 1][0] = sy1;
    featurevector[2 * k + 1][1] = sx1;
    featurevector[2 * k + 1][2] = 0;
    featurevector[2 * k + 1][3] = 1;
    featurevector[2 * k + 1][4] = -sy2;
  }

  /////////////////////////////////////////////////////////////
  // Least square distance method
  ////////////////////////////////////////////////////////////
  double w[5];
  double v[5][5];

  int minwindex = 0;
  double minw = NAN;

  svdcmp(featurevector, w, v);

  minwindex = 0;
  minw = w[0];
  for (k = 1; k < 5; k++) {
    if (minw > w[k]) {
      minw = w[k];
      minwindex = k;
    }
  }

  Hmatrix[0][0] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[0][1] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[0][2] = (float)(v[2][minwindex] / v[4][minwindex]);
  Hmatrix[1][0] = (float)(v[1][minwindex] / v[4][minwindex]);
  Hmatrix[1][1] = (float)(v[0][minwindex] / v[4][minwindex]);
  Hmatrix[1][2] = (float)(v[3][minwindex] / v[4][minwindex]);
  Hmatrix[2][0] = 0.0f;
  Hmatrix[2][1] = 0.0f;
  Hmatrix[2][2] = 1.0f;
}