#include <iostream>
#include <iomanip>
#include <cmath>
#include <gsl/gsl_sf_erf.h>
#include <unordered_map>
#include <functional>

#define EPS 3.0e-11
#define N 28
#define LB -3
#define RB 3

enum phif
{
  phif_erf,
};

static float xsav,ysav;
//static float x[N], w[N];

static float x28[] = {0,-2.9893274307250977,-2.9439094066619873,-2.8627779483795166,
                      -2.7468991279602051,-2.59767746925354,-2.416923999786377,
                      -2.2068326473236084,-1.9699532985687256,-1.7091614007949829,
                      -1.4276226758956909,-1.1287544965744019,-0.8161848783493042,
                      -0.49370783567428589,-0.16523787379264832,0.16523787379264832,
                      0.49370783567428589,0.8161848783493042,1.1287544965744019,
                      1.4276226758956909,1.7091614007949829,1.9699532985687256,
                      2.2068326473236084,2.416923999786377,2.59767746925354,
                      2.7468991279602051,2.8627779483795166,2.9439094066619873,};
static float w28[] = {2.9893274307250977,0.027372848242521286,0.063396334648132324,
                      0.09870428591966629,0.132818803191185,0.16532203555107117,
                      0.19581876695156097,0.22393864393234253,0.2493402510881424,
                      0.27171522378921509,0.29079198837280273,0.30633890628814697,
                      0.3181672990322113,0.32613357901573181,0.33014103770256042,
                      0.33014103770256042,0.32613357901573181,0.3181672990322113,
                      0.30633890628814697,0.29079198837280273,0.27171522378921509,
                      0.2493402510881424,0.22393864393234253,0.19581876695156097,
                      0.16532203555107117,0.132818803191185,0.09870428591966629,
                      0.063396334648132324};

static float x28_2[] = {0, -3.487548828125, -3.434561014175415, -3.3399074077606201, -3.2047154903411865, -3.0306239128112793, -2.8197448253631592, -2.5746381282806396, -2.29827880859375, -1.9940216541290283, -1.6655597686767578, -1.3168803453445435, -0.95221567153930664, -0.57599246501922607, -0.19277751445770264, 0.19277751445770264, 0.57599246501922607, 0.95221567153930664, 1.3168803453445435, 1.6655597686767578, 1.9940216541290283, 2.29827880859375, 2.5746381282806396, 2.8197448253631592, 3.0306239128112793, 3.2047154903411865, 3.3399074077606201, 3.434561014175415};
static float w28_2[] = {3.487548828125, 0.031934987753629684, 0.073962390422821045, 0.11515499651432037, 0.15495526790618896, 0.19287571310997009, 0.22845523059368134, 0.26126176118850708, 0.2908969521522522, 0.3170011043548584, 0.33925729990005493, 0.35739538073539734, 0.37119516730308533, 0.38048917055130005, 0.38516455888748169, 0.38516455888748169, 0.38048917055130005, 0.37119516730308533, 0.35739538073539734, 0.33925729990005493, 0.3170011043548584, 0.2908969521522522, 0.26126176118850708, 0.22845523059368134, 0.19287571310997009, 0.15495526790618896, 0.11515499651432037, 0.073962390422821045};


static float x56[] = {0, -3.4968302249908447, -3.483309268951416, -3.4590280055999756, -3.4240560531616211, -3.3785006999969482, -3.322502613067627, -3.2562351226806641, -3.1799025535583496, -3.0937414169311523, -2.9980175495147705, -2.8930270671844482, -2.7790942192077637, -2.6565713882446289, -2.5258374214172363, -2.387296199798584, -2.2413759231567383, -2.0885276794433594, -1.929223895072937, -1.7639570236206055, -1.5932378768920898, -1.4175940752029419, -1.2375686168670654, -1.0537178516387939, -0.86661016941070557, -0.67682385444641113, -0.48494547605514526, -0.29156816005706787, -0.097289621829986572, 0.097289621829986572, 0.29156816005706787, 0.48494547605514526, 0.67682385444641113, 0.86661016941070557, 1.0537178516387939, 1.2375686168670654, 1.4175940752029419, 1.5932378768920898, 1.7639570236206055, 1.929223895072937, 2.0885276794433594, 2.2413759231567383,2.387296199798584, 2.5258374214172363, 2.6565713882446289, 2.7790942192077637, 2.8930270671844482, 2.9980175495147705, 3.0937414169311523, 3.1799025535583496, 3.2562351226806641, 3.322502613067627,3.3785006999969482, 3.4240560531616211, 3.4590280055999756, 3.483309268951416};
static float w56[] = {3.4968302249908447, 0.0081334933638572693, 0.018908828496932983, 0.029641721397638321, 0.040284384042024612, 0.050802811980247498, 0.061164293438196182, 0.071336753666400909, 0.081288732588291168, 0.090989455580711365, 0.10040894150733948, 0.10951806604862213, 0.11828868836164474, 0.12669368088245392, 0.13470706343650818, 0.14230409264564514, 0.14946125447750092, 0.15615645051002502, 0.16236896812915802, 0.16807961463928223, 0.17327073216438293, 0.17792628705501556, 0.18203188478946686, 0.18557482957839966, 0.18854416906833649, 0.19093072414398193, 0.19272713363170624, 0.19392783939838409, 0.19452911615371704, 0.19452911615371704, 0.19392783939838409,0.19272713363170624, 0.19093072414398193, 0.18854416906833649, 0.18557482957839966, 0.18203188478946686, 0.17792628705501556, 0.17327073216438293, 0.16807961463928223, 0.16236896812915802, 0.15615645051002502, 0.14946125447750092, 0.14230409264564514, 0.13470706343650818, 0.12669368088245392, 0.11828868836164474, 0.10951806604862213, 0.10040894150733948, 0.090989455580711365, 0.081288732588291168, 0.071336753666400909, 0.061164293438196182, 0.050802811980247498, 0.040284384042024612, 0.029641721397638321, 0.018908828496932983};


std::unordered_map<float,float> results;

#include <math.h>
float erfcc(float x) {
  // License prevents distribution of source code
}

float erff(float x) {
  //return 1.0 - erfcc(x);
  std::unordered_map<float,float>::iterator ii = results.find(x);
  if (ii != results.end()) {
    return ii->second;
  } else {
    return results[x] = 1.0-erfcc(x);
  }
}

// Compute coefficients for Gaussian quadrature 
void gauleg(float x1, float x2, float x[], float w[], int n) {
  // License prevents distribution of source code
}

float phi0(float x, float mu, float sigma) {
  return 0.5 * (1 + erff((x - mu) / (sqrt(2) * sigma)));
};


// ------------------------------------------------------
// ----- C_integrate_0 ----------------------------------
// ------------------------------------------------------

float inner_0(float x, float a, std::function <float (float)> phi) {
  return 1/sqrt(2*M_PI) * exp((-x*x)/2) * phi( a*x );
}

float qgaus28_0(float (*func)(float, float, std::function <float (float)> phi), float a, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, phi));
  }
  return s *= xr;
}

float qgaus28_2_0(float (*func)(float, float, std::function <float (float)> phi), float a, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28_2[j];
      s += w28_2[j]*((*func)(xm+dx, a, phi));
  }
  return s *= xr;
}

float qgaus56_0(float (*func)(float, float, std::function <float (float)> phi), float a, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<56;j++) {
      dx=xr*x56[j];
      s += w56[j]*((*func)(xm+dx, a, phi));
  }
  return s *= xr;
}

float C0(float a, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  //return qgaus28_0(&inner_0, a, phi);
  return qgaus28_2_0(&inner_0, a, phi);
  //return qgaus56_0(&inner_0, a, phi);
}

// ------------------------------------------------------
// ----- C_integrate_0_1 ----------------------------------
// ------------------------------------------------------

float inner_0_1(float x, float a, float b, std::function <float (float)> phi) {
  return 1/sqrt(2*M_PI) * exp((-x*x)/2) * phi( a*x + b );
}

float qgaus28_0_1(float (*func)(float, float, float, std::function <float (float)> phi), float a, float b, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, b, phi));
  }
  return s *= xr;
}

float qgaus56_0_1(float (*func)(float, float, float, std::function <float (float)> phi), float a, float b, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<56;j++) {
      dx=xr*x56[j];
      s += w56[j]*((*func)(xm+dx, a, b, phi));
  }
  return s *= xr;
}

float C0_1(float a, float b, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  return qgaus28_0_1(&inner_0_1, a, b, phi);
  //return qgaus56_0_1(&inner_0_1, a, b, phi);
}

// ------------------------------------------------------
// ----- C_integrate_1 ----------------------------------
// ------------------------------------------------------

float inner_1(float x, float y, float a, float b, std::function <float (float)> phi) {
  return std::pow(1/sqrt(2*M_PI), 2) * exp((-x*x-y*y)/2) * //1;
         phi( a*x ) * 
         phi( b*y );
}

float qgaus28_1(float (*func)(float, float, float, std::function <float (float)> phi), float a, float b, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, b, phi));
  }
  return s *= xr;
}

float qgaus56_1(float (*func)(float, float, float, std::function <float (float)> phi), float a, float b, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<56;j++) {
      dx=xr*x56[j];
      s += w56[j]*((*func)(xm+dx, a, b, phi));
  }
  return s *= xr;
}

float f1_1(float x, float a, float b, std::function <float (float)> phi) {
  float f2_1(float y, float a, float b, std::function <float (float)> phi);
  xsav = x;
  return qgaus28_1(f2_1, a, b, phi);
  //return qgaus56_1(f2_1, a, b, phi);
}

float f2_1(float y, float a, float b, std::function <float (float)> phi) {
  return (*inner_1)(xsav, y, a, b, phi);
}

float C1(float a, float b, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  return qgaus28_1(&f1_1, a, b, phi);
  //return qgaus56_1(&f1_1, a, b, phi);
}

// ------------------------------------------------------
// ----- C_integrate_2 ----------------------------------
// ------------------------------------------------------

float inner_2(float x, float y, float z, float a, float b, float c, float d, std::function <float (float)> phi) {
  return std::pow(1/sqrt(2*M_PI), 3) * exp((-x*x-y*y-z*z)/2) * //1;
         phi( a*x + c*z ) * 
         phi( b*y + d*z );
}

float qgaus28_2(float (*func)(float, float, float, float, float, std::function <float (float)> phi), float a, float b, float c, float d, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, b, c, d, phi));
  }
  return s *= xr;
}

float qgaus28_2_2(float (*func)(float, float, float, float, float, std::function <float (float)> phi), float a, float b, float c, float d, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28_2[j];
      s += w28_2[j]*((*func)(xm+dx, a, b, c, d, phi));
  }
  return s *= xr;
}

float qgaus56_2(float (*func)(float, float, float, float, float, std::function <float (float)> phi), float a, float b, float c, float d, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<56;j++) {
      dx=xr*x56[j];
      s += w56[j]*((*func)(xm+dx, a, b, c, d, phi));
  }
  return s *= xr;
}

float f1_2(float x, float a, float b, float c, float d, std::function <float (float)> phi) {
  float f2_2(float y, float a, float b, float c, float d, std::function <float (float)> phi);
  xsav = x;
  //return qgaus28_2(f2_2, a, b, c, d, phi);
  return qgaus56_2(f2_2, a, b, c, d, phi);
}

float f2_2(float y, float a, float b, float c, float d, std::function <float (float)> phi) {
  float f3_2(float z, float a, float b, float c, float d, std::function <float (float)> phi);
  ysav = y;
  //return qgaus28_2(f3_2, a, b, c, d, phi);
  return qgaus56_2(f3_2, a, b, c, d, phi);
}

float f3_2(float z, float a, float b, float c, float d, std::function <float (float)> phi) {
  return (*inner_2)(xsav, ysav, z, a, b, c, d, phi);
}

float C2(float a, float b, float c, float d, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  //return qgaus28_2(&f1_2, a, b, c, d, phi);
  //return qgaus56_2(&f1_2, a, b, c, d, phi);
  return qgaus28_2_2(&f1_2, a, b, c, d, phi);
}

// ------------------------------------------------------
// ----- C_integrate_3 ----------------------------------
// ------------------------------------------------------

float inner_3(float x, float y, float a, float b, float c, std::function <float (float)> phi) {
  return std::pow(1/sqrt(2*M_PI), 2) * exp((-x*x-y*y)/2) * 
         phi( a*x ) * 
         phi( b*x + c*y );
}

float qgaus28_3(float (*func)(float, float, float, float, std::function <float (float)> phi), float a, float b, float c, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, b, c, phi));
  }
  return s *= xr;
}

float qgaus28_2_3(float (*func)(float, float, float, float, std::function <float (float)> phi), float a, float b, float c, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28_2[j];
      s += w28_2[j]*((*func)(xm+dx, a, b, c, phi));
  }
  return s *= xr;
}

float qgaus56_3(float (*func)(float, float, float, float, std::function <float (float)> phi), float a, float b, float c, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<56;j++) {
      dx=xr*x56[j];
      s += w56[j]*((*func)(xm+dx, a, b, c, phi));
  }
  return s *= xr;
}

float f1_3(float x, float a, float b, float c, std::function <float (float)> phi) {
  float f2_3(float y, float a, float b, float c, std::function <float (float)> phi);
  xsav = x;
  return qgaus28_3(f2_3, a, b, c, phi);
  //return qgaus56_3(f2_3, a, b, c, phi);
}

float f2_3(float y, float a, float b, float c, std::function <float (float)> phi) {
  return (*inner_3)(xsav, y, a, b, c, phi);
}

float C3(float a, float b, float c, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  //return qgaus28_3(&f1_3, a, b, c, phi);
  //return qgaus28_2_3(&f1_3, a, b, c, phi);
  return qgaus56_3(&f1_3, a, b, c, phi);
}

// ------------------------------------------------------
// ----- C_integrate_4 ----------------------------------
// ------------------------------------------------------

float inner_4(float x, float a, float b, float c, float d, std::function <float (float)> phi) {
  return 1/sqrt(2*M_PI) * exp((-x*x)/2) * 
         phi( a*x + c ) * 
         phi( b*x + d );
}

float qgaus28_4(float (*func)(float, float, float, float, float, std::function <float (float)> phi), float a, float b, float c, float d, std::function <float (float)> phi) 
{
  int j;
  float s;
  float xr,xm,dx;

  xm=0.5*(RB+LB);
  xr=0.5*(RB-LB);
  s=0;

  for (j=1;j<28;j++) {
      dx=xr*x28[j];
      s += w28[j]*((*func)(xm+dx, a, b, c, d, phi));
  }
  return s *= xr;
}

float C4(float a, float b, float c, float d, float mu, float sigma, int phif) {
  std::function <float (float)> phi;
  switch(phif) {
    case phif_erf : phi = std::bind(&phi0, std::placeholders::_1, mu, sigma);
                    break;
    default       : throw std::invalid_argument("Invalid transfer function. Exiting now.\n");
  }
  return qgaus28_4(&inner_4, a, b, c, d, phi);
}

int main() {
  //float mu = 0.22;
  //float sigma = 0.1;
  //float a = 0.0;
  //float b = 0.0;
  //float c = 1;
  //float d = 1;
  //float result1 = C3(0.4, 0.4, 0.4, mu, sigma, phif_erf);
  //std::cout << std::setprecision (17) << "Result: " << result1 << std::endl;
  //float result2 = C_2(a, b, c, d, mu, sigma, phif_erf);
  //std::cout << std::setprecision (17) << "Result: " << result << std::endl;
  //float result3 = C_3(a, b, c, d, mu, sigma, phif_erf);
  //std::cout << std::setprecision (17) << "Result: " << result << std::endl;
  int n = 28;
  float x1 = -3.5;
  float x2 = 3.5;
  float x[n], w[n];
  x[0] = 0;
  gauleg(x1, x2, x, w, n);
  std::cout << "Points" << std::endl;
  for (int i=0;i<n;i++) {
     std::cout << std::setprecision (17) << x[i] << ", ";// << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Weights" << std::endl;
  for (int i=0;i<n;i++) {
     std::cout << std::setprecision (17) << w[i] << ", ";
  }
}
