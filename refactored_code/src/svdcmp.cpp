// ------------------------------------------------------------------------
// An implementation of SVD from Numerical Recipes in C and Mike's lectures
// ------------------------------------------------------------------------

#include "svdcmp.h"

#include <math.h>
#include <stdio.h>

static double maxarg1, maxarg2;
#define FMAX(a, b) \
  (maxarg1 = (a), maxarg2 = (b), (maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static int iminarg1, iminarg2;
#define IMIN(a, b)                 \
  (iminarg1 = (a), iminarg2 = (b), \
   (iminarg1 < (iminarg2) ? (iminarg1) : iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)

// ------------------------------------------------------------------------
// calculates sqrt( a^2 + b^2 ) with decent precision
// ------------------------------------------------------------------------

double pythag(double a, double b) {
  double absa, absb;

  absa = fabs(a);
  absb = fabs(b);

  if (absa > absb)
    return (absa * sqrt(1.0 + SQR(absb / absa)));
  else
    return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}
