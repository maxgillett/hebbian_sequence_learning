#include <math.h> 

#define SQPI sqrt(4.*atan(1.))
#define TWOPI (8.*atan(1.))
#define a1  -1.26551223
#define a2   1.00002368
#define a3   0.37409196
#define a4   0.09678418
#define a5  -0.18628806
#define a6   0.27886087
#define a7  -1.13520398
#define a8   1.48851587
#define a9  -0.82215223
#define a10  0.17087277

double nerf(double z) {
  double t,ef,at;
  double w;
  w = fabs(z);
  t = 1.0e0/(1.0e0 + 0.5e0 * w);
  at=a1+t*(a2+t*(a3+t*(a4+t*(a5+t*(a6+t*(a7+t*(a8+t*(a9+t*a10))))))));
  ef=t*exp(at);
  if(z>0.0e0)
    ef = 2.0e0*exp(w*w)-ef;  
  return(ef);
}

/**** static transfer function ****/

double Phi(double mu,double sigma,double taum,double threshold,double reset,double taurp) {
  double w,x,y,z,cont,ylow;
  int i,N=10000;
  x=(threshold-mu)/sigma;
  y=(reset-mu)/sigma;
  w=0.;
  if(x<-100.&&y<-100.) {
    w=log(y/x)-0.25/pow(x,2.)+0.25/pow(y,2.);
    w=1./(taurp+taum*w);
  }
  else if(y<-100.) {
    ylow=-100.; 
    N=(int)(100.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(double)(i)/(double)(N);
      cont=nerf(z);
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*SQPI/(double)(N);
    w+=log(-y/100.)-0.000025+0.25/pow(y,2.);
    w=1./(taurp+taum*w);
  }
  else {
    ylow=y;
    N=(int)(100.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(double)(i)/(double)(N);
      cont=nerf(z);
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*SQPI/(double)(N);
    w=1./(taurp+taum*w);
  }
  return(w);
}
