// ------------------------------------------------------------------------
// An implementation of SVD from Numerical Recipes in C and Mike's lectures
// ------------------------------------------------------------------------

#pragma once

#include <stdio.h>

#include <cmath>
#include <vector>
#include <array>
#include <cstdint>

template <typename T>
auto SIGN(T a, T b) -> T {
  return ((b) > 0.0 ? std::abs(a) : -std::abs(a));
}

// ------------------------------------------------------------------------
// calculates sqrt( a^2 + b^2 ) with decent precision
// ------------------------------------------------------------------------
double pythag(double a, double b);

// ------------------------------------------------------------------------
// Modified from Numerical Recipes in C
// Given a matrix a[nRows][nCols], svdcmp() computes its singular value
// decomposition, A = U * W * Vt.  A is replaced by U when svdcmp
// returns.  The diagonal matrix W is output as a vector w[nCols].
// V (not V transpose) is output as the matrix V[nCols][nCols].
// CAUTION : Output is unsorted!!!!!
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Modified from Numerical Recipes in C
// Given a matrix a[nRows][nCols], svdcmp() computes its singular value
// decomposition, A = U * W * Vt.  A is replaced by U when svdcmp
// returns.  The diagonal matrix W is output as a vector w[nCols].
// V (not V transpose) is output as the matrix V[nCols][nCols].
// ------------------------------------------------------------------------
template <size_t nCols>
inline int svdcmp(std::vector<std::array<double, nCols>>& a, double (&w)[nCols],
                  double (&v)[nCols][nCols]) {
  const size_t nRows = a.size();
  int flag, its;
  size_t jj, j, nm;
  intmax_t i, k, l;
  double anorm, c, f, g, h, s, scale, x, y, z;

  double rv1[nCols];

  g = scale = anorm = 0.0;
  for (i = 0; i < static_cast<intmax_t>(nCols); i++) {
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < static_cast<intmax_t>(nRows)) {
      for (k = i; k < static_cast<intmax_t>(nRows); k++) scale += std::abs(a[k][i]);
      if (scale) {
        for (k = i; k < static_cast<intmax_t>(nRows); k++) {
          a[k][i] /= scale;
          s += a[k][i] * a[k][i];
        }
        f = a[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][i] = f - g;
        for (j = l; j < nCols; j++) {
          for (s = 0.0, k = i; k < static_cast<intmax_t>(nRows); k++) s += a[k][i] * a[k][j];
          f = s / h;
          for (k = i; k < static_cast<intmax_t>(nRows); k++) a[k][j] += f * a[k][i];
        }
        for (k = i; k < static_cast<intmax_t>(nRows); k++) a[k][i] *= scale;
      }
    }
    w[i] = scale * g;
    g = s = scale = 0.0;
    if (i < static_cast<intmax_t>(nRows) && i != nCols - 1) {
      for (k = l; k < static_cast<intmax_t>(nCols); k++) scale += std::abs(a[i][k]);
      if (scale) {
        for (k = l; k < static_cast<intmax_t>(nCols); k++) {
          a[i][k] /= scale;
          s += a[i][k] * a[i][k];
        }
        f = a[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][l] = f - g;
        for (k = l; k < static_cast<intmax_t>(nCols); k++) rv1[k] = a[i][k] / h;
        for (j = l; j < nRows; j++) {
          for (s = 0.0, k = l; k < static_cast<intmax_t>(nCols); k++) s += a[j][k] * a[i][k];
          for (k = l; k < static_cast<intmax_t>(nCols); k++) a[j][k] += s * rv1[k];
        }
        for (k = l; k < static_cast<intmax_t>(nCols); k++) a[i][k] *= scale;
      }
    }
    anorm = std::max(anorm, (std::abs(w[i]) + std::abs(rv1[i])));

    fflush(stdout);
  }

  for (i = nCols - 1; i >= 0; i--) {
    if (i < static_cast<intmax_t>(nCols - 1)) {
      if (g) {
        for (j = l; j < nCols; j++) v[j][i] = (a[i][j] / a[i][l]) / g;
        for (j = l; j < static_cast<intmax_t>(nCols); j++) {
          for (s = 0.0, k = l; k < static_cast<intmax_t>(nCols); k++) s += a[i][k] * v[k][j];
          for (k = l; k < static_cast<intmax_t>(nCols); k++) v[k][j] += s * v[k][i];
        }
      }
      for (j = l; j < nCols; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
    fflush(stdout);
  }

  for (i = std::min(nRows, nCols) - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    for (j = l; j < nCols; j++) a[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      for (j = l; j < nCols; j++) {
        for (s = 0.0, k = l; k < static_cast<intmax_t>(nRows); k++) s += a[k][i] * a[k][j];
        f = (s / a[i][i]) * g;
        for (k = i; k < static_cast<intmax_t>(nRows); k++) a[k][j] += f * a[k][i];
      }
      for (j = i; j < nRows; j++) a[j][i] *= g;
    } else
      for (j = i; j < nRows; j++) a[j][i] = 0.0;
    ++a[i][i];
    fflush(stdout);
  }

  for (k = nCols - 1; k >= 0; k--) {
    for (its = 0; its < 30; its++) {
      flag = 1;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if ((std::abs(rv1[l]) + anorm) == anorm) {
          flag = 0;
          break;
        }
        if ((std::abs(w[nm]) + anorm) == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          rv1[i] = c * rv1[i];
          if ((std::abs(f) + anorm) == anorm) break;
          g = w[i];
          h = pythag(f, g);
          w[i] = h;
          h = 1.0 / h;
          c = g * h;
          s = -f * h;
          for (j = 0; j < nRows; j++) {
            y = a[j][nm];
            z = a[j][i];
            a[j][nm] = y * c + z * s;
            a[j][i] = z * c - y * s;
          }
        }
      }
      z = w[k];
      if (l == k) {
        if (z < 0.0) {
          w[k] = -z;
          for (j = 0; j < nCols; j++) v[j][k] = -v[j][k];
        }
        break;
      }
      if (its == 29) printf("no convergence in 30 svdcmp iterations\n");
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = pythag(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = pythag(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y *= c;
        for (jj = 0; jj < nCols; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = x * c + z * s;
          v[jj][i] = z * c - x * s;
        }
        z = pythag(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
        for (jj = 0; jj < nRows; jj++) {
          y = a[jj][j];
          z = a[jj][i];
          a[jj][j] = y * c + z * s;
          a[jj][i] = z * c - y * s;
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
    fflush(stdout);
  }

  return (0);
}

// ------------------------------------------------------------------------
// Modified from Numerical Recipes in C
// Given a matrix a[nRows][nCols], svdcmp() computes its singular value
// decomposition, A = U * W * Vt.  A is replaced by U when svdcmp
// returns.  The diagonal matrix W is output as a vector w[nCols].
// V (not V transpose) is output as the matrix V[nCols][nCols].
// CAUTION : Output is unsorted!!!!!
// ------------------------------------------------------------------------
int svdcmp(double **a, int nRows, int nCols, double *w, double **v,
           double *rv1);
