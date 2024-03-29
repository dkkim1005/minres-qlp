#ifndef MINRESQLP_CPP
#define MINRESQLP_CPP

/*

 MINRESQLP C++ implementation based on the original code,
 minresqlpModule.f90(09 Sep 2013)

  * Original author and contributor:
     Author:
	Sou-Cheng Choi <sctchoi@uchicago.edu>
	Computation Institute (CI)
	University of Chicago
	Chicago, IL 60637, USA

	Michael Saunders <saunders@stanford.edu>
	Systems Optimization Laboratory (SOL)
	Stanford University
	Stanford, CA 94305-4026, USA

     Contributor:
	Christopher Paige <paige@cs.mcgill.ca>

  Searching for detailed descriptions, see http://web.stanford.edu/group/SOL/software/minresqlp/

  /-------------- Development info --------------
       Date : 27 april 2018
    Version : 0.0.0
  Developer : Dongkyu Kim <dkkim1005@gist.ac.kr>
  ----------------------------------------------/
  
*/


#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>
#include <iomanip>

using REAL = double;
using IMAG = std::complex<double>;

namespace MINRESQLP
{
// NumberType : REAL / IMAG 
template<typename NumberType>
struct BaseInfo
{
  typedef std::vector<NumberType> ContainerType;
  BaseInfo(const int n_,
 		   const ContainerType& b_,
		   const double shift_     = 0,
		   const bool   useMsolve_ = false,
		   const bool   disable_   = false,
		   const int    itnlim_    = -1,
		   const double rtol_      = 1e-16,
		   const double maxxnorm_  = 1e7,
		   const double trancond_  = 1e7,
		   const double Acondlim_  = 1e15,
		   const bool   print_     = false);
  virtual void Aprod(const int n, const NumberType *x, NumberType *y) const = 0;
  virtual void Msolve(const int n, const NumberType *x, NumberType *y) const
  {
    if(useMsolve)
	  std::cerr << " & Warning! the method Msolve is not overridden." << std::endl;
  }
  // inputs
  int n, itnlim;
  ContainerType b;
  double shift, rtol, maxxnorm, trancond, Acondlim;
  bool   useMsolve, disable, print;
  // outputs
  ContainerType x;
  int istop, itn;
  double rnorm, Arnorm, xnorm, Anorm, Acond;
};


template<typename NumberType>
BaseInfo<NumberType>::BaseInfo(const int n_, const ContainerType& b_,
  const double shift_, const bool   useMsolve_,
  const bool   disable_, const int    itnlim_,
  const double rtol_, const double maxxnorm_,
  const double trancond_, const double Acondlim_,
  const bool   print_):
  // initialize list for inputs
  n(n_),
  b(b_),
  itnlim(itnlim_<0 ? 4*n_ : itnlim_),
  shift(shift_),
  useMsolve(useMsolve_),
  disable(disable_),
  rtol(rtol_),
  maxxnorm(maxxnorm_),
  trancond(trancond_),
  Acondlim(Acondlim_),
  print(print_),
  // initialize list for outputs
  x(n_),
  istop(0),
  itn(0),
  rnorm(0),
  Arnorm(0),
  xnorm(0),
  Acond(0) {}


template<typename INFO_t>
class RealSolver
{
  typedef std::vector<double> dvector;
public:
  void solve(INFO_t& client) const;
private:
  void symortho_(const double& a, const double& b, double &c, double &s, double &r) const;
  double dnrm2_(const int n, const double* x, const int incx) const;
  void printstate_(const int iter,       const double x1,     const double xnorm,
			       const double rnorm,   const double Arnorm, const double relres,
			       const double relAres, const double Anorm,  const double Acond) const;
  static constexpr double eps_ = std::numeric_limits<double>::epsilon();
};

template<typename INFO_t>
void RealSolver<INFO_t>::symortho_(const double& a, const double& b, double &c, double &s, double &r) const
{
	double t, abs_a = std::abs(a), abs_b = std::abs(b);

	if (abs_b <= eps_) {
		s = 0;
		r = abs_a;
		if (a == 0) {
			c = 1;
		} else {
			c = a/abs_a;
		}
	} else if (abs_a <= eps_) {
		c = 0;
		r = abs_b;
		s = b / abs_b;
	} else if (abs_b > abs_a) {
		t = a / b;
		s = (b / abs_b) / std::sqrt(1. + std::pow(t, 2));
		c = s*t;
		r = b/s;
	} else {
		t = b/a;
		c = (a/abs_a)/std::sqrt(1. + std::pow(t, 2));
		s = c*t;
		r = a/c;
	}
}

template<typename INFO_t>
double RealSolver<INFO_t>::dnrm2_(const int n, const double* x, const int incx) const
{
	int ix;
	double ssq, absxi, norm, scale;

	if (n<1 || incx < 1) {
		norm = 0;
	} else if (n==1) {
		norm = std::abs(x[0]);
	} else {
		scale = 0, ssq = 1;

		for (ix=0; ix<(1+(n-1)*incx); ix += incx) {
			if (x[ix] != 0) {
				absxi = std::abs(x[ix]);
				if (scale < absxi) {
					ssq = 1. + ssq*std::pow(scale/absxi, 2);
					scale = absxi;
				} else {
					ssq += std::pow(absxi/scale, 2);
				}
			}
		}
		norm = scale*std::sqrt(ssq);
	}

	return norm;
}

template<typename INFO_t>
void RealSolver<INFO_t>::printstate_(const int iter,       const double x1,     const double xnorm,
	        	        const double rnorm,   const double Arnorm, const double relres,
		                const double relAres, const double Anorm,  const double Acond) const
{
	std::cout << std::setw(7) << "iter "
		  << std::setw(14) << "x[0] "
		  << std::setw(14) << "xnorm "
		  << std::setw(14) << "rnorm "
		  << std::setw(14) << "Arnorm "
		  << std::setw(14) << "Compatible "
		  << std::setw(14) << "LS "
		  << std::setw(14) << "norm(A)"
		  << std::setw(14) << "cond(A)"
		  << std::endl;

	std::cout << std::setprecision(7)
		  << std::setw(6)  << iter
		  << std::setw(14) << x1
		  << std::setw(14) << xnorm
		  << std::setw(14) << rnorm
		  << std::setw(14) << Arnorm
		  << std::setw(14) << relres
		  << std::setw(14) << relAres
		  << std::setw(14) << Anorm
		  << std::setw(14) << Acond
		  << "\n\n"
		  << std::flush;
}

template<typename INFO_t>
void RealSolver<INFO_t>::solve(INFO_t& client) const
{
	const int n = client.n;
	const dvector &b = client.b, zero(n,0);
	dvector& x = client.x;

	// local constants
	const double EPSINV  = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10))),
		     NORMMAX = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10)/2.));

	// local arrays and variables
	const double shift_ = client.shift;
	const bool   checkA_ = true, precon_ = client.useMsolve, disable_ = client.disable;
	double rtol_ = client.rtol,
           maxxnorm_ = std::min({client.maxxnorm, 1./eps_}),
   	   trancond_ = std::min({client.trancond, NORMMAX}),
	   Acondlim_ = std::min({client.Acondlim, EPSINV});

	double Arnorm_ = 0, xnorm_ = 0, Anorm_ = 0, Acond_ = 1;
	int itnlim_ = client.itnlim, istop_ = 0, itn_ = 0;

	dvector r1(n), r2(n), v(n), w(n), wl(n), wl2(n),
                xl2(n), y(n), vec2(2), vec3(3);

	double Axnorm    = 0,  beta      = 0,        beta1     = dnrm2_(n, &b[0], 1),
	       betan     = 0,  ieps      = 0.1/eps_, pnorm     = 0,
	       relAres   = 0,  relAresl  = 0,        relresl   = 0,     
           t1        = 0,  t2        = 0,        xl2norm   = 0,
	       cr1       =-1,  cr2       =-1,        cs        =-1,
	       dbar      = 0,  dltan     = 0,        epln      = 0,
	       eplnn     = 0,  eta       = 0,        etal      = 0,
           etal2     = 0,  gama      = 0,        gama_QLP  = 0,
	       gamal     = 0,  gamal_QLP = 0,        gamal_tmp = 0,
	       gamal2    = 0,  gamal3    = 0,        gmin      = 0,
           gminl     = 0,  phi       = 0,        s         = 0,
           sn        = 0,  sr1       = 0,        sr2       = 0,
           t         = 0,  tau       = 0,        taul      = 0,
	       taul2     = 0,  u         = 0,        u_QLP     = 0,          
           ul        = 0,  ul_QLP    = 0,        ul2       = 0,
	       ul3       = 0,  ul4       = 0,        vepln     = 0,
           vepln_QLP = 0,  veplnl    = 0,        veplnl2   = 0,
	       x1last    = 0,  xnorml    = 0,        Arnorml   = 0,
	       Anorml    = 0,  rnorml    = 0,        Acondl    = 0;

    int QLPiter = 0, flag0 = 0;
	bool done = false, lastiter = false, likeLS;

    const std::vector<std::string> msg = {
         "beta_{k+1} < eps.                                                ", //  1
         "beta2 = 0.  If M = I, b and x are eigenvectors of A.             ", //  2
         "beta1 = 0.  The exact solution is  x = 0.                        ", //  3
         "A solution to (poss. singular) Ax = b found, given rtol.         ", //  4
         "A solution to (poss. singular) Ax = b found, given eps.          ", //  5
         "Pseudoinverse solution for singular LS problem, given rtol.      ", //  6
         "Pseudoinverse solution for singular LS problem, given eps.       ", //  7
         "The iteration limit was reached.                                 ", //  8
         "The operator defined by Aprod appears to be unsymmetric.         ", //  9
         "The operator defined by Msolve appears to be unsymmetric.        ", //  10
         "The operator defined by Msolve appears to be indefinite.         ", //  11
         "xnorm has exceeded maxxnorm or will exceed it next iteration.    ", //  12
         "Acond has exceeded Acondlim or 0.1/eps.                          ", //  13
         "Least-squares problem but no converged solution yet.             ", //  14
         "A null vector obtained, given rtol.                              "};//  15

    x = zero, xl2 = zero;

	if (client.print)
	{
		std::cout << std::setprecision(3);
		std::cout << std::endl
			  << std::setw(54) << "Enter MINRES-QLP(INFO)" << std::endl
			  << "  "
			  << "\n\n"
			  << std::setw(14) << "n  = "        << std::setw(8) << n
			  << std::setw(14) << "||b||  = "    << std::setw(8) << beta1
			  << std::setw(14) << "precon  = "   << std::setw(8) << ((precon_) ? "true" : "false")
			  << std::endl
			  << std::setw(14) << "itnlim  = "   << std::setw(8) << itnlim_
			  << std::setw(14) << "rtol  = "     << std::setw(8) << rtol_
			  << std::setw(14) << "shift  = "    << std::setw(8) << shift_
			  << std::endl
			  << std::setw(14) << "raxxnorm  = " << std::setw(8) << maxxnorm_
			  << std::setw(14) << "Acondlim  = " << std::setw(8) << Acondlim_
			  << std::setw(14) << "trancond  = " << std::setw(8) << trancond_
			  << std::endl
			  << "  "
			  << std::endl << std::endl
			  << std::flush;
	}

	y = b, r1 = b;

	if (precon_) client.Msolve(n, &b[0], &y[0]);

	beta1 = std::inner_product(b.begin(), b.end(), y.begin(), 0.0);

	if (beta1 < 0 && dnrm2_(n, &y[0], 1) > eps_) istop_ = 11;

	if (beta1 == 0) istop_ = 3;

	beta1 = std::sqrt(beta1);

	if (checkA_ && precon_) {
		client.Msolve(n, &y[0], &r2[0]);
		s = std::inner_product(y.begin(), y.end(), y.begin(), 0.0),
		t = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0);
		double z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
		if (z > epsa) istop_ = 10;
	}

	if (checkA_) {
		client.Aprod(n, &y[0], &w[0]);
		client.Aprod(n, &w[0], &r2[0]);
		s = std::inner_product(w.begin(), w.end(), w.begin(), 0.0),
		t = std::inner_product(y.begin(), y.end(), r2.begin(), 0.0);
		double z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
		if (z > epsa) istop_ = 9;
	}

	betan = beta1, phi = beta1;
    	double rnorm_ = betan;
    	double relres = rnorm_ / (Anorm_*xnorm_ + beta1);
    	r2 = b, w = zero, wl = zero, done = false;

	// MINRESQLP iteration loop.
	while(istop_ <= flag0) {
		itn_ += 1;
		double betal = beta;
		beta = betan;
		s = 1./beta;
		for(int index=0; index<n; ++index) v[index] = s*y[index];
		client.Aprod(n, &v[0], &y[0]);
		if (shift_ != 0) {
			for(int index=0; index<n; ++index) y[index] -= shift_*v[index];
		}
		if (itn_ >= 2) {
			for(int index=0; index<n; ++index) y[index] += (-beta/betal)*r1[index];
		}

		double alfa = std::inner_product(v.begin(), v.end(), y.begin(), 0.0);
		for(int index=0; index<n; index++) y[index] = y[index] + (-alfa/beta)*r2[index];
		r1 = r2, r2 = y;

		if (!precon_) {
			betan = dnrm2_(n, &y[0], 1);
		} else {
			client.Msolve(n, &r2[0], &y[0]);
			betan = std::inner_product(r2.begin(), r2.end(), y.begin(), 0.0);
			if (betan > 0) {
				betan = std::sqrt(betan);
			} else if (dnrm2_(n, &y[0], 1) > eps_) {
				istop_ = 11;
				break;
			}
		}

		if (itn_ == 1) {
			vec2[0] = alfa, vec2[1] = betan;
			pnorm = dnrm2_(2, &vec2[0], 1);
		} else {
			vec3[0] = beta, vec3[1] = alfa, vec3[2] = betan;
			pnorm = dnrm2_(3, &vec3[0], 1);
		}

		dbar = dltan;
		double dlta = cs*dbar + sn*alfa;
		epln = eplnn;
		double gbar = sn*dbar - cs*alfa;
		eplnn = sn*betan, dltan = -cs*betan;
		double dlta_QLP = dlta;

		gamal3 = gamal2, gamal2 = gamal, gamal  = gama;

		symortho_(gbar, betan, cs, sn, gama);
		double gama_tmp = gama;
		taul2 = taul;
		taul = tau;
		tau = cs*phi;
		phi = sn*phi;
		Axnorm = std::sqrt(std::pow(Axnorm,2) + std::pow(tau,2));

       		// apply the previous right reflection P{k-2,k}
		if (itn_ > 2) {
          		veplnl2  = veplnl;
          		etal2    = etal;
          		etal     = eta;
          		double dlta_tmp = sr2 * vepln - cr2 * dlta;
          		veplnl   = cr2 * vepln + sr2 * dlta;
          		dlta     = dlta_tmp;
          		eta      = sr2 * gama;
          		gama     = -cr2 * gama;
		}

       		// compute the current right reflection P{k-1,k}, P_12, P_23,...
		if (itn_ > 1) {
          		symortho_(gamal, dlta, cr1, sr1, gamal_tmp);
          		gamal     = gamal_tmp;
          		vepln     = sr1 * gama;
          		gama      = -cr1 * gama;
		}

       		// update xnorm
                double xnorml = xnorm_;
       		ul4    = ul3;
       		ul3    = ul2;

		if (itn_ > 2) ul2 = ( taul2 - etal2 * ul4 - veplnl2 * ul3 ) / gamal2;
		if (itn_ > 1) ul  = ( taul  - etal  * ul3 - veplnl  * ul2) / gamal;

       		vec3[0] = xl2norm, vec3[1] = ul2, vec3[2] = ul;

       		double xnorm_tmp = dnrm2_(3, &vec3[0], 1);  // norm([xl2norm ul2 ul]);

       		if (std::abs(gama) > eps_) {
          		u = (tau - eta*ul2 - vepln*ul) / gama;
     			likeLS  = relAresl < relresl;
		        vec2[0] = xnorm_tmp, vec2[1] = u;
          		if (likeLS && dnrm2_(2, &vec2[0], 1) > maxxnorm_) {
             			u = 0;
             			istop_ = 12;
			}
		} else {
			u = 0;
			istop_ = 14;
		}

		vec2[0] = xl2norm, vec2[1] = ul2;
       		xl2norm = dnrm2_(2, &vec2[0], 1);
       		vec3[0] = xl2norm, vec3[1] = ul, vec3[2] = u;
       		xnorm_  = dnrm2_(3, &vec3[0], 1);

		// MINRES updates
        	if (Acond_ < trancond_ && istop_ == flag0 && QLPiter == 0) {
			wl2 = wl;
			wl = w;
			if (gama_tmp > eps_) {
				s = 1./gama_tmp;
				for (int index=0; index<n; ++index) w[index] = (v[index] - epln*wl2[index] - dlta_QLP*wl[index])*s;
			}

          		if (xnorm_ < maxxnorm_) {
             			x1last = x[0];
				for (int index=0; index<n; ++index) x[index] += tau*w[index];
			} else {
             			istop_ = 12, lastiter = true;
			}
		} else {
          		QLPiter += 1;
          		if (QLPiter == 1) {
             			xl2 = zero;
             			if (itn_ > 1) { // construct w_{k-3}, w_{k-2}, w_{k-1}
				
                			if (itn_ > 3) {
						for (int index=0; index<n; ++index) {
							wl2[index] = gamal3*wl2[index] + veplnl2*wl[index] + etal*w[index];
						}
					}
                			if (itn_ > 2) {
						for (int index=0; index<n; ++index) {
							wl[index] = gamal_QLP*wl[index] + vepln_QLP*w[index];
						}
					}

					for (int index=0; index<n; ++index)  w[index] *= gama_QLP;
					
					for (int index=0; index<n; ++index) {
                				xl2[index] = x[index] - ul_QLP*wl[index] - u_QLP*w[index];
					}
				}
			}
		
          	if (itn_ == 1) {
             	wl2 =  wl;
				for (int index=0; index<n; ++index) wl[index]  =  sr1*v[index];
				for (int index=0; index<n; ++index) w[index]   = -cr1*v[index];
			} else if (itn_ == 2) {
             			wl2 = wl;
				for (int index=0; index<n; ++index) wl[index]  = cr1*w[index] + sr1*v[index];
				for (int index=0; index<n; ++index) w[index]   = sr1*w[index] - cr1*v[index];
			} else {
             	wl2 = wl;
             	wl  = w;
				for (int index=0; index<n; ++index) w[index]   = sr2*wl2[index] - cr2*v[index];
				for (int index=0; index<n; ++index) wl2[index] = cr2*wl2[index] + sr2*v[index];
				for (int index=0; index<n; ++index) v[index]   = cr1*wl[index]  + sr1*w[index];
				for (int index=0; index<n; ++index) w[index]   = sr1*wl[index]  - cr1*w[index];
             	wl  = v;
			}
          	x1last = x[0];
			for (int index=0; index<n; ++index) xl2[index] = xl2[index] + ul2*wl2[index];
			for (int index=0; index<n; ++index) x[index]   = xl2[index] + ul *wl[index] + u*w[index];
		}

       	// compute the next right reflection P{k-1,k+1}
       	gamal_tmp = gamal;
       	symortho_(gamal_tmp, eplnn, cr2, sr2, gamal);
       	// store quantities for transfering from MINRES to MINRESQLP
       	gamal_QLP = gamal_tmp;
       	vepln_QLP = vepln;
       	gama_QLP  = gama;
       	ul_QLP    = ul;
       	u_QLP     = u;

       	// estimate various norms
       	double abs_gama = abs(gama);
		Anorml = Anorm_;
       	Anorm_ = std::max({Anorm_, pnorm, gamal, abs_gama});

       	if (itn_ == 1) {
          		gmin  = gama;
          		gminl = gmin;
		} else if (itn_ > 1) {
          		double gminl2  = gminl;
          		gminl   = gmin;
          		vec3[0] = gminl2, vec3[1] = gamal, vec3[2] = abs_gama;
          		gmin    = std::min({gminl2, gamal, abs_gama});
		}

       		double Acondl = Acond_;
       		Acond_ = Anorm_ / gmin;
       		double rnorml   = rnorm_;
       		relresl = relres;

       		if (istop_ != 14) rnorm_ = phi;
       		relres = rnorm_ / (Anorm_ * xnorm_ + beta1);
       		vec2[0] = gbar, vec2[1] = dltan;
       		double rootl = dnrm2_(2, &vec2[0], 1);
		Arnorml  = rnorml * rootl;
       		relAresl = rootl / Anorm_;

       		// see if any of the stopping criteria are satisfied.
       		double epsx = Anorm_*xnorm_*eps_;

       		if (istop_ == flag0 || istop_ == 14) {
          		t1 = 1. + relres, t2 = 1. + relAresl;
		}
		if (t1 <= 1) {
			istop_ = 5;
		} else if (t2 <= 1) {
			istop_ = 7;
		} else if (relres <= rtol_) {
			istop_ = 4;
		} else if (relAresl <= rtol_) {
			istop_ = 6;
		} else if (epsx >= beta1) {
			istop_ = 2;
		} else if (xnorm_ >= maxxnorm_) {
			istop_ = 12;
		} else if (Acond_ >= Acondlim_ || Acond_ >= ieps) {
			istop_ = 13;
		} else if (itn_ >= itnlim_) {
			istop_ = 8;
		} else if (betan < eps_) {
			istop_ = 1;
		}

		if (disable_ && itn_ < itnlim_) {
			istop_ = flag0, done = false;
			if (Axnorm < rtol_*Anorm_*xnorm_) {
				istop_ = 15, lastiter = false;
			}
		}

		if (istop_ != flag0) {
			done = true;
			if (istop_ == 6 || istop_ == 7 || istop_ == 12 || istop_ == 13) lastiter = true;
			if (lastiter) itn_ -= 1, Acond_ = Acondl, rnorm_ = rnorml, relres = relresl;
			
			client.Aprod(n, &x[0], &r1[0]);
			for (int index=0; index<n; ++index) r1[index] = b[index] - r1[index] + shift_*x[index];
			client.Aprod(n, &r1[0], &wl2[0]);
			for (int index=0; index<n; ++index) wl2[index] = wl2[index] - shift_*r1[index];
          		Arnorm_ = dnrm2_(n, &wl2[0], 1);
          		if (rnorm_ > 0 && Anorm_ > 0) relAres = Arnorm_ / (Anorm_*rnorm_);
		}

		if(client.print) printstate_(itn_-1, x1last, xnorml, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl);
		
	}

	// end of iteration loop.
        client.istop  =  istop_, client.itn   =   itn_, client.rnorm = rnorm_;
	client.Arnorm = Arnorm_, client.xnorm = xnorm_, client.Anorm = Anorm_;
        client.Acond  = Acond_;

	if (client.print) {
		printstate_(itn_, x[0], xnorm_, rnorm_, Arnorm_, relres, relAres, Anorm_, Acond_);
		std::cout << "  " << "Exit MINRES-QLP" << ": "
			  << msg[istop_-1] << "\n\n";
	}
}


// Part for hermitian matrix solver 

typedef std::complex<double> dcomplex;
typedef std::vector<dcomplex> zvector;

template<typename INFO_t>
class HermitianSolver
{
public:
  void solve(INFO_t& client) const;
private:
  void zsymortho_(const dcomplex& a, const dcomplex& b, double& c, dcomplex& s, dcomplex& r) const;
  dcomplex zdotc_(const int n, const dcomplex* cx, const int incx, const dcomplex* cy, const int incy) const;
  double znrm2_(const int n, const dcomplex* x, const int incx) const;
  void printstate_(const int iter,       const dcomplex x1,   const double xnorm,
			       const double rnorm,   const double Arnorm, const double relres,
			       const double relAres, const double Anorm,  const double Acond) const;
  static constexpr double eps_ = std::numeric_limits<double>::epsilon();
  static constexpr double realmin_ = std::numeric_limits<double>::min();
};

template<typename INFO_t>
void HermitianSolver<INFO_t>::printstate_(const int iter,       const dcomplex x1,   const double xnorm,
			             const double rnorm,   const double Arnorm, const double relres,
			             const double relAres, const double Anorm,  const double Acond) const
{
	std::cout << std::setw(7) << "iter "
		  << std::setw(21) << "x[0] "
		  << std::setw(14) << "xnorm "
		  << std::setw(14) << "rnorm "
		  << std::setw(14) << "Arnorm "
		  << std::setw(14) << "Compatible "
		  << std::setw(14) << "LS "
		  << std::setw(14) << "norm(A)"
		  << std::setw(14) << "cond(A)"
		  << std::endl;

	std::cout << std::setprecision(4)
		  << std::setw(6)  << iter
		  << std::setw(21) << x1
		  << std::setw(14) << xnorm
		  << std::setw(14) << rnorm
		  << std::setw(14) << Arnorm
		  << std::setw(14) << relres
		  << std::setw(14) << relAres
		  << std::setw(14) << Anorm
		  << std::setw(14) << Acond
		  << "\n\n"
		  << std::flush;
}

template<typename INFO_t>
dcomplex HermitianSolver<INFO_t>::zdotc_(const int n, const dcomplex* cx,
	       const int incx, const dcomplex* cy, const int incy) const
{
	dcomplex ctemp = dcomplex(0,0);
	int ix, iy;

	if (n <= 0) return dcomplex(0,0);

	if (incx == 1 && incy == 1) {
		for(int i = 0; i<n; ++i) ctemp += std::conj(cx[i])*cy[i];
	} else {
    		if (incx >= 0) {
			ix = 1;
		} else {
			ix = (-n + 1)*incx + 1;
		}

    		if (incy >= 0) {
			iy = 1;
		} else {
			iy = (-n + 1)*incy + 1;
		}

		for(int i=0; i<n; ++i) {
			ctemp += std::conj(cx[ix])*cy[iy];
			ix += incx;
			iy += incy;
		}
	}

	return ctemp;
}

template<typename INFO_t>
double HermitianSolver<INFO_t>::znrm2_(const int n, const dcomplex* x, const int incx) const
{
	double norm, scale, ssq;

	if (n< 1 || incx < 1) {
		norm = 0;
	} else {
		scale = 0, ssq = 1;

		for (int ix = 0; ix<1+(n - 1)*incx; ix += incx) {
			if (x[ix].real() != 0) {
				double temp = std::abs(x[ix].real());
        			if (scale < temp) {
					ssq = 1 + ssq*std::pow(scale/temp, 2);
					scale = temp;
				} else {
					ssq += std::pow(temp/scale, 2);
				}
			}
	
			if (x[ix].imag()!= 0) {
        			double temp = std::abs(x[ix].imag());
        			if (scale < temp) {
          				ssq = 1 + ssq*std::pow(scale/temp, 2);
					scale = temp;
				} else {
					ssq += std::pow(temp / scale, 2);
				}
			}
		}

    		norm  = scale*std::sqrt (ssq);
	}

	return norm;
}

template<typename INFO_t>
void HermitianSolver<INFO_t>::zsymortho_(const dcomplex& a, const dcomplex& b, double& c, dcomplex& s, dcomplex& r) const
{
	double t, abs_a = std::abs(a), abs_b = std::abs(b);

	if (abs_b <= realmin_) {
		c = 1., s = dcomplex(0,0), r = a;
	} else if (abs_a <= realmin_) {
		c = 0, s = 1, r = b;
	} else if (abs_b > abs_a) {
		t = abs_a/abs_b;
		c = 1./std::sqrt(1. + std::pow(t,2));
		s = c*std::conj((b/abs_b) / (a/abs_a));
		c = c*t;
		r = b/std::conj(s);
	} else {
		t = abs_b/abs_a;
		c = 1. / std::sqrt(1. + std::pow(t,2));
		s = c*t*std::conj((b/abs_b) / (a/abs_a));
		r = a/c;
	}
}


template<typename INFO_t>
void HermitianSolver<INFO_t>::solve(INFO_t& client) const
{
  const int n = client.n;
  const zvector &b = client.b, zero(n,0);
  zvector& x = client.x;
  // local arrays and variables
  double shift_, rtol_, maxxnorm_, trancond_, Acondlim_,
         rnorm_, Arnorm_, xnorm_ = 0, Anorm_ = 0, Acond_ = 1.;
  bool   checkA_, precon_, disable_;
  int    itnlim_, nout_, istop_, itn_ = 0;
  zvector r1(n), r2(n), v(n), w(n), wl(n), wl2(n), xl2(n), y(n), vec2(2), vec3(3);
  double  Axnorm   = 0, beta    = 0, gmin    = 0,
          gminl    = 0, pnorm   = 0, relAres = 0,
          relAresl = 0, relresl = 0, t1      = 0,
          t2       = 0, xl2norm = 0, cr1     =-1,
          cr2      = -1, cs     =-1;

  dcomplex dbar      = dcomplex(0,0), dltan     = dcomplex(0,0), eplnn     = dcomplex(0,0),
           eta       = dcomplex(0,0), etal      = dcomplex(0,0), etal2     = dcomplex(0,0),
           gama      = dcomplex(0,0), gama_QLP  = dcomplex(0,0), gamal     = dcomplex(0,0),
           gamal_QLP = dcomplex(0,0), gamal_tmp = dcomplex(0,0), gamal2    = dcomplex(0,0),
           s         = dcomplex(0,0), sn        = dcomplex(0,0), sr1       = dcomplex(0,0),
           sr2       = dcomplex(0,0), t         = dcomplex(0,0), tau       = dcomplex(0,0),
           taul      = dcomplex(0,0), u         = dcomplex(0,0), u_QLP     = dcomplex(0,0),
           ul        = dcomplex(0,0), ul_QLP    = dcomplex(0,0), ul2       = dcomplex(0,0),
           ul3       = dcomplex(0,0), vepln     = dcomplex(0,0), vepln_QLP = dcomplex(0,0),
           veplnl    = dcomplex(0,0), veplnl2   = dcomplex(0,0), x1last    = x[0];

  int QLPiter = 0, flag0 = 0;
  bool done, lastiter, likeLS;

  // local constants
  const double EPSINV  = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10))),
               NORMMAX = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10)/2.));

  const std::vector<std::string> msg = {
         "beta_{k+1} < eps.                                                ", // 1
         "beta2 = 0.  If M = I, b and x are eigenvectors of A.             ", // 2
         "beta1 = 0.  The exact solution is  x = 0.                        ", // 3
         "A solution to (poss. singular) Ax = b found, given rtol.         ", // 4
         "A solution to (poss. singular) Ax = b found, given eps.          ", // 5
         "Pseudoinverse solution for singular LS problem, given rtol.      ", // 6
         "Pseudoinverse solution for singular LS problem, given eps.       ", // 7
         "The iteration limit was reached.                                 ", // 8
         "The operator defined by Aprod appears to be non-Hermitian.       ", // 9
         "The operator defined by Msolve appears to be non-Hermitian.      ", // 10
         "The operator defined by Msolve appears to be indefinite.         ", // 11
         "xnorm has exceeded maxxnorm  or will exceed it next iteration.   ", // 12
         "Acond has exceeded Acondlim or 0.1/eps.                          ", // 13
         "Least-squares problem but no converged solution yet.             ", // 14
         "A null vector obtained, given rtol.                              "};// 15

  shift_    = client.shift,
  checkA_   = true;
  disable_  = client.disable;
  itnlim_   = client.itnlim;
  rtol_     = client.rtol;
  maxxnorm_ = std::min({client.maxxnorm, 1./eps_});
  trancond_ = std::min({client.trancond, NORMMAX});
  Acondlim_ = client.Acondlim;
  precon_   = client.useMsolve;
  lastiter  = false;

  istop_   = flag0;
  double beta1 = znrm2_(n, &b[0], 1), ieps     = 0.1/eps_;

  x        = zero;
  xl2      = zero;
  x1last   = x[0];

  y  = b;
  r1 = b;

  if (precon_)
    client.Msolve(n, &b[0], &y[0]);


  beta1 = (zdotc_(n, &b[0], 1, &y[0], 1)).real();

  if (beta1<0 && znrm2_(n, &y[0], 1)>eps_)
	istop_ = 11;

  if (beta1 == 0)
	istop_ = 3;

  beta1 = std::sqrt(beta1);

  if (checkA_ && precon_)
  {
    client.Msolve(n, &y[0], &r2[0]);
    s = zdotc_(n, &y[0], 1, &y[0], 1);
    t = zdotc_(n, &r1[0], 1, &r2[0], 1);
    double z = std::abs(s-t), epsa = (std::abs(s) + eps_) * std::pow(eps_, 0.33333);
    if (z > epsa)
	  istop_ = 10;
  }

  if (checkA_)
  {
    client.Aprod(n, &y[0], &w[0]);
    client.Aprod(n, &w[0], &r2[0]);
    s = zdotc_(n, &w[0], 1, &w[0], 1);
    t = zdotc_(n, &y[0], 1, &r2[0], 1);
    double z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
    if (z > epsa)
	  istop_ = 9;
  }

  double betan  = beta1;
  dcomplex phi    = dcomplex(beta1, 0);
  rnorm_ = betan;
  double relres = rnorm_ / (Anorm_*xnorm_ + beta1);
  relAres= 0;
  r2     = b;
  w      = zero;
  wl     = zero;
  done   = false;

  if (client.print)
  {
    std::cout << std::setprecision(3);
    std::cout << std::endl
              << std::setw(54) << "Enter MINRES-QLP(INFO)" << std::endl
              << "  "
              << "\n\n"
              << std::setw(14) << "n  = "        << std::setw(8) << n
              << std::setw(14) << "||b||  = "    << std::setw(8) << beta1
              << std::setw(14) << "precon  = "   << std::setw(8) << ((precon_) ? "true" : "false")
              << std::endl
              << std::setw(14) << "itnlim  = "   << std::setw(8) << itnlim_
              << std::setw(14) << "rtol  = "     << std::setw(8) << rtol_
              << std::setw(14) << "shift  = "    << std::setw(8) << shift_
              << std::endl
              << std::setw(14) << "raxxnorm  = " << std::setw(8) << maxxnorm_
              << std::setw(14) << "Acondlim  = " << std::setw(8) << Acondlim_
              << std::setw(14) << "trancond  = " << std::setw(8) << trancond_
              << std::endl
              << "  "
              << std::endl << std::endl
              << std::flush;
  }

  // main iteration loop.
  while (istop_ <= flag0)
  {
    itn_ += 1; // k = itn = 1 first time through
    double betal  = beta;               // betal = betak
    beta   = betan;
    s      = 1./beta;         // Normalize previous vector (in y).
    for (int index=0; index<n; ++index)
	  v[index] = s*y[index]; // v = vk if P = I.

    client.Aprod(n, &v[0], &y[0]);

    if (std::abs(shift_) >= realmin_)
      for (int index=0; index<n; ++index)
		y[index] += -shift_*v[index];


    if (itn_ >= 2) 
      for (int index=0; index<n; ++index)
		y[index] += (-beta/betal)*r1[index];
    

    double alfa = (zdotc_(n, &v[0], 1, &y[0], 1)).real();
    for (int index=0; index<n; ++index)
	  y[index] += (-alfa/beta)*r2[index];
    r1 = r2;
    r2 = y;

    if (!precon_) 
      betan = znrm2_(n, &y[0], 1);
    else
	{
      client.Msolve(n, &r2[0], &y[0]);
      betan = (zdotc_(n, &r2[0], 1, &y[0], 1)).real();
      if (betan > 0)
	  {
        betan = std::sqrt(betan);
      }
	  else if (znrm2_(n, &y[0], 1) > eps_)
	  { // M must be indefinite.
        istop_ = 11;
        break;
	  }
    }

    if (itn_ == 1)
    {
      vec2[0] = alfa, vec2[1] = betan;
      pnorm   = znrm2_(2, &vec2[0], 1);
    }
    else
    {
      vec3[0] = beta, vec3[1] = alfa, vec3[2] = betan;
      pnorm   = znrm2_(3, &vec3[0], 1);
    }

    // Apply previous left reflection Qk-1 to get
    //   [deltak epslnk+1] = [cs  sn][dbark    0   ]
    //   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

    dbar   = dltan;
    dcomplex dlta   = cs * dbar  +  sn * alfa;  // dlta1  = 0         deltak
    dcomplex epln   = eplnn;
    dcomplex gbar   = sn * dbar  -  cs * alfa;  // gbar 1 = alfa1     gbar k
             eplnn  =               sn * betan; // eplnn2 = 0         epslnk+1
             dltan  =            -  cs * betan; // dbar 2 = beta2     dbar k+1
		dcomplex dlta_QLP = dlta;

    // Compute the current left reflection Qk
    dcomplex gamal3 = gamal2;
    gamal2 = gamal;
    gamal  = gama;
    zsymortho_(gbar, dcomplex(betan,0), cs, sn, gama);

    dcomplex gama_tmp = gama;
    dcomplex taul2  = taul;
    taul   = tau;
    tau    = cs * phi;
    phi    = sn * phi;                   //  phik
    Axnorm = std::sqrt( std::pow(Axnorm,2) + std::pow(std::abs(tau), 2));

    // Apply the previous right reflection P{k-2,k}
    if (itn_ > 2)
	{
      veplnl2  = veplnl;
      etal2    = etal;
      etal     = eta;
      dcomplex dlta_tmp = sr2 * vepln - cr2 * dlta;
      veplnl   = cr2 * vepln + sr2 * dlta;
      dlta     = dlta_tmp;
      eta      = sr2 * gama;
      gama     = -cr2 * gama;
    }

    // Compute the current right reflection P{k-1,k}, P_12, P_23,...
    if (itn_ > 1)
    {
      zsymortho_(gamal, dlta, cr1, sr1, gamal_tmp);
      gamal     = gamal_tmp;
      vepln     = sr1 * gama;
      gama      = -cr1 * gama;
    }

    // Update xnorm
    double xnorml  = xnorm_;
    dcomplex ul4 = ul3;
    ul3     = ul2;

    if (itn_ > 2)
	  ul2 = ( taul2 - etal2 * ul4 - veplnl2 * ul3 ) / gamal2;

    if (itn_ > 1)
	  ul  = ( taul  - etal  * ul3 - veplnl  * ul2) / gamal;

    vec3[0] = xl2norm, vec3[1] = ul2, vec3[2] = ul;
    double xnorm_tmp = znrm2_(3, &vec3[0], 1);  // norm([xl2norm ul2 ul]);

    if (std::abs(gama) > 0  &&  xnorm_tmp < maxxnorm_)
    {
      u = (tau - eta*ul2 - vepln*ul) / gama;
      likeLS  = relAresl < relresl;
      vec2[0] = xnorm_tmp;
      vec2[1] = u;
      if (likeLS && znrm2_(2, &vec2[0], 1) > maxxnorm_)
	  {
        u      = 0;
        istop_ = 12;
      }
    }
    else
    {
      u      = 0;
      istop_ = 14;
    }

    vec2[0]   = xl2norm, vec2[1]   = ul2;
    xl2norm   = znrm2_(2, &vec2[0], 1);
    vec3[0]   = xl2norm, vec3[1]   = ul, vec3[2]   = u;
    xnorm_    = znrm2_(3, &vec3[0], 1);

    if (Acond_ < trancond_ && istop_ == flag0 && QLPiter == 0) // MINRES updates
    { 
      wl2  = wl;
      wl   = w;
      if (std::abs(gama_tmp) > 0)
      {
        s = 1. / gama_tmp;
		for(int index=0; index<n; ++index)
          w[index] = (v[index] - epln*wl2[index] - dlta_QLP*wl[index])*s;
      }
      if (xnorm_ < maxxnorm_)
      {
        x1last = x[0];
        for(int index=0; index<n; ++index)
          x[index] = x[index] + tau*w[index];
      }
      else
      {
        istop_   = 12;
        lastiter = true;
      }
    }
	else //MINRES-QLP updates
	{ 
      QLPiter += 1;
      if (QLPiter == 1)
      {
        xl2 = zero; // vector
        if (itn_ > 1)
        {
          // construct w_{k-3}, w_{k-2}, w_{k-1}
          if (itn_ > 3)
          {
            for (int index=0; index<n; ++index)
              wl2[index] = gamal3*wl2[index] + veplnl2*wl[index] + etal*w[index];
		  } // w_{k-3}
		  if (itn_ > 2)
          {
            for (int index=0; index<n; ++index)
              wl[index] = gamal_QLP*wl[index] + vepln_QLP*w[index];
          } // w_{k-2}
          for (int index=0; index<n; ++index)
			w[index] = gama_QLP*w[index];
          for (int index=0; index<n; ++index)
            xl2[index] =  x[index] - ul_QLP*wl[index] - u_QLP*w[index];
        }
      }

      if (itn_ == 1)
      {
        wl2 =  wl;
        for (int index=0; index<n; ++index)
          wl[index]  =  sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index]   = -cr1*v[index];
      }
      else if (itn_ == 2)
      {
        wl2 = wl;
        for (int index=0; index<n; ++index)
          wl[index]  = cr1*w[index] + sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*w[index] - cr1*v[index];
      }
      else
      {
        wl2 = wl;
        wl  = w;
        for (int index=0; index<n; ++index)
          w[index] = sr2*wl2[index] - cr2*v[index];
        for (int index=0; index<n; ++index)
          wl2[index] = cr2*wl2[index] + sr2*v[index];
        for (int index=0; index<n; ++index)
          v[index] = cr1*wl[index] + sr1*w[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*wl[index] - cr1*w[index];
        wl  = v;
      }

      x1last = x[0];
      for (int index=0; index<n; ++index)
        xl2[index] += ul2*wl2[index];
      for (int index=0; index<n; ++index)
        x[index] = xl2[index] + ul*wl[index] + u*w[index];
    }

    // Compute the next right reflection P{k-1,k+1}
    gamal_tmp = gamal;
    zsymortho_(gamal_tmp, eplnn, cr2, sr2, gamal);

    // Store quantities for transfering from MINRES to MINRES-QLP
    gamal_QLP = gamal_tmp;
    vepln_QLP = vepln;
    gama_QLP  = gama;
    ul_QLP    = ul;
    u_QLP     = u;

    // Estimate various norms
    double abs_gama = std::abs(gama);
    double Anorml   = Anorm_;
    Anorm_ = std::max({Anorm_, pnorm, abs(gamal), abs_gama});

    if (itn_ == 1)
    {
      gmin  = abs_gama;
      gminl = gmin;
    }
    else if (itn_ > 1)
    {
      double gminl2  = gminl;
      gminl = gmin;
      vec3[0] = gminl2, vec3[1] = gamal, vec3[2] = abs_gama;
      gmin = std::min({gminl2, std::abs(gamal), abs_gama});
    }

    double Acondl = Acond_;
    Acond_ = Anorm_ / gmin;
    double rnorml = rnorm_;
    relresl = relres;

    if (istop_ != 14)
      rnorm_ = std::abs(phi);

    relres = rnorm_ / (Anorm_ * xnorm_ + beta1);
    vec2[0]  = gbar;
    vec2[1]  = dltan;
    double rootl = znrm2_(2, &vec2[0], 1);
    double Arnorml = rnorml * rootl;
    relAresl = rootl / Anorm_;

    // See if any of the stopping criteria are satisfied.
    double epsx = Anorm_*xnorm_*eps_;
    if (istop_ == flag0 || istop_ == 14)
    {
      t1 = 1. + relres;
      t2 = 1. + relAresl;
    }

    if (t1 <= 1.)
      istop_ = 5;                           // Accurate Ax=b solution
	else if (t2 <= 1)
      istop_ = 7;                           // Accurate LS solution
	else if (relres   <= rtol_)
	  istop_ = 4;                           // Good enough Ax=b solution
	else if (relAresl <= rtol_)
      istop_ = 6;                           // Good enough LS solution
	else if (epsx >= beta1)
      istop_ = 2;                           // x is an eigenvector
	else if (xnorm_ >= maxxnorm_)
      istop_ = 12;                          // xnorm exceeded its limit
    else if (Acond_ >= Acondlim_ || Acond_ >= ieps)
      istop_ = 13;                          // Huge Acond
    else if (itn_ >= itnlim_)
      istop_ = 8;                           // Too many itns
    else if (betan < eps_)
      istop_ = 1;                           // Last iteration of Lanczos

    if (disable_ && itn_ < itnlim_)
	{
      istop_ = flag0;
      done   = false;
      if (Axnorm < rtol_*Anorm_*xnorm_)
      {
        istop_   = 15;
        lastiter = false;
      }
    }

    if (istop_ != flag0)
    {
      done = true;
      if (istop_ == 6 || istop_ == 7 || istop_ == 12 || istop_ == 13)
        lastiter = true;
      if (lastiter)
	  {
        itn_ -= 1;
        Acond_ = Acondl;
        rnorm_ = rnorml;
        relres = relresl;
      }

      client.Aprod(n, &x[0], &r1[0]);
      for (int index=0; index<n; ++index)
        r1[index]  = b[index] - r1[index] + shift_*x[index]; // r1 to temporarily store residual vector
      client.Aprod(n, &r1[0], &wl2[0]); // wl2 to temporarily store A*r1
      for (int index=0; index<n; ++index)
        wl2[index] = wl2[index] - shift_*r1[index];
      Arnorm_ = znrm2_(n, &wl2[0], 1);
      if (rnorm_ > 0 && Anorm_ > 0)
        relAres = Arnorm_ / (Anorm_*rnorm_);
    }

    if (client.print)
    {
      if (itn_ <= 11 || itn_%10 == 1)
      {
        printstate_(itn_-1, x1last, xnorml, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl);
        if (itn_ == 11)
		  std::cout << std::endl;
      }
    }

	if (istop_ != flag0)
      break;

  } // end of iteration loop.

  client.istop = istop_, client.itn = itn_, client.rnorm = rnorm_;
  client.Arnorm = Arnorm_, client.xnorm = xnorm_, client.Anorm = Anorm_;
  client.Acond = Acond_;

  if (client.print)
  {
    printstate_(itn_, x[0], xnorm_, rnorm_, Arnorm_, relres, relAres, Anorm_, Acond_);
    std::cout << "  " << "Exit MINRES-QLP" << ": "
			  << msg[istop_-1] << "\n\n";
  }
}

} // end namespace minresqlp

#endif
