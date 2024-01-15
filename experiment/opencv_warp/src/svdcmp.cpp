// ------------------------------------------------------------------------
// An implementation of SVD from Numerical Recipes in C and Mike's lectures
// ------------------------------------------------------------------------

#include "svdcmp.h"

#include <cmath>
#include <cstdio>

double SQR(double a) { return a == 0.0 ? 0.0 : a * a; }

// ------------------------------------------------------------------------
// calculates sqrt( a^2 + b^2 ) with decent precision
// ------------------------------------------------------------------------

double pythag(double a, double b) {
  const double absa = std::abs(a);
  const double absb = std::abs(b);

  if (absa > absb)
    return (absa * sqrt(1.0 + SQR(absb / absa)));
  else
    return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}
