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
#include <algorithm>
#include <numeric>
#include <iomanip>

// Ref : https://github.com/gon1332/fort320/blob/master/include/Utils/colors.h
#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif  /* _COLORS_ */

namespace minresqlp {

struct baseInfo {
	typedef std::vector<double> container_t;

	baseInfo(const int n_,
		 const container_t& b_,
		 const double shift_     = 0,
		 const bool   useMsolve_ = false,
		 const bool   disable_   = false,
		 const int    itnlim_    = -1,
		 const double rtol_      = 1e-16,
		 const double maxxnorm_  = 1e7,
		 const double trancond_  = 1e7,
		 const double Acondlim_  = 1e15)

	: // initialize list for inputs
	  n(n_), b(b_), itnlim(itnlim_<0 ? 4*n_ : itnlim_),
	  shift(shift_), useMsolve(useMsolve_), disable(disable_), rtol(rtol_),
	  maxxnorm(maxxnorm_), trancond(trancond_), Acondlim(Acondlim_),

	  // initialize list for outputs
	  x(n_), istop(0), itn(0), rnorm(0),
	  Arnorm(0), xnorm(0), Acond(0) {}

	virtual void
	Aprod(const int n, const double *x, double *y) const = 0;

	virtual void
	Msolve(const int n, const double *x, double *y) const {
		if(!useMsolve) {
			std::cout << " --- Error! Msolve is not overridden..." << std::endl;
			exit(1);
		}
	};
 
	// inputs
	const int n, itnlim;
	const container_t b;
	const double shift, rtol, maxxnorm, trancond, Acondlim;
	const bool   useMsolve, disable;

	// outputs
	container_t x;
	int istop, itn;
	double rnorm, Arnorm, xnorm, Anorm, Acond;
};

template<typename INFO_t>
class mainsolver {
	typedef std::vector<double> dvector;
	public:
		void
		solve(INFO_t& client, const bool print = false) const;
	private:
		void
		_symortho(const double& a, const double& b, double &c, double &s, double &r) const;

		double
		_dnrm2(const int n, const double* x, const int incx) const;

		void
		_printstate(const int iter,       const double x1,     const double xnorm,
			    const double rnorm,   const double Arnorm, const double relres,
			    const double relAres, const double Anorm,  const double Acond) const;

		static constexpr double _eps = std::numeric_limits<double>::epsilon();
};

template<typename INFO_t>
void
mainsolver<INFO_t>::_symortho(const double& a, const double& b, double &c, double &s, double &r) const {
	double t, abs_a = std::abs(a), abs_b = std::abs(b);

	if (abs_b <= _eps) {
		s = 0;
		r = abs_a;
		if (a == 0) {
			c = 1;
		} else {
			c = a/abs_a;
		}
	} else if (abs_a <= _eps) {
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
double
mainsolver<INFO_t>::_dnrm2(const int n, const double* x, const int incx) const {
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
void
mainsolver<INFO_t>::_printstate(const int iter,       const double x1,     const double xnorm,
	        	        const double rnorm,   const double Arnorm, const double relres,
		                const double relAres, const double Anorm,  const double Acond) const {

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
		  << "\n\n";
}




template<typename INFO_t>
void
mainsolver<INFO_t>::solve(INFO_t& client, const bool print) const {
	const int n = client.n;
	const dvector &b = client.b, zero(n,0);
	dvector& x = client.x;

	// local constants
	const double EPSINV  = std::pow(10.0, std::floor(std::log(1./_eps)/std::log(10))),
		     NORMMAX = std::pow(10.0, std::floor(std::log(1./_eps)/std::log(10)/2.));

	// local arrays and variables
	const double shift_ = client.shift;
	const bool   checkA_ = true, precon_ = client.useMsolve, disable_ = client.disable;
	double rtol_ = client.rtol,
           maxxnorm_ = std::min({client.maxxnorm, 1./_eps}),
   	   trancond_ = std::min({client.trancond, NORMMAX}),
	   Acondlim_ = std::min({client.Acondlim, EPSINV});

	double Arnorm_ = 0, xnorm_ = 0, Anorm_ = 0, Acond_ = 1;
	int itnlim_ = client.itnlim, istop_ = 0, itn_ = 0;

	dvector r1(n), r2(n), v(n), w(n), wl(n), wl2(n),
                xl2(n), y(n), vec2(2), vec3(3);

	double Axnorm    = 0,  beta      = 0,        beta1     = _dnrm2(n, &b[0], 1),
	       betan     = 0,  ieps      = 0.1/_eps, pnorm     = 0,
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
         FBLU("beta_{k+1} < eps.                                                "), //  1
         FBLU("beta2 = 0.  If M = I, b and x are eigenvectors of A.             "), //  2
         BOLD(FBLU("beta1 = 0.  The exact solution is  x = 0.                        ")), //  3
         FGRN("A solution to (poss. singular) Ax = b found, given rtol.         "), //  4
         BOLD(FGRN("A solution to (poss. singular) Ax = b found, given eps.          ")), //  5
         FGRN("Pseudoinverse solution for singular LS problem, given rtol.      "), //  6
         BOLD(FGRN("Pseudoinverse solution for singular LS problem, given eps.       ")), //  7
         FYEL("The iteration limit was reached.                                 "), //  8
         BOLD(FRED("The operator defined by Aprod appears to be unsymmetric.         ")), //  9
         BOLD(FRED("The operator defined by Msolve appears to be unsymmetric.        ")), //  10
         BOLD(FRED("The operator defined by Msolve appears to be indefinite.         ")), //  11
         FYEL("xnorm has exceeded maxxnorm or will exceed it next iteration.    "), //  12
         FYEL("Acond has exceeded Acondlim or 0.1/eps.                          "), //  13
         FRED("Least-squares problem but no converged solution yet.             "), //  14
         FYEL("A null vector obtained, given rtol.                              ")};//  15

    	x = zero, xl2 = zero;

	if (print) {
		std::cout << std::setprecision(3);
		std::cout << std::endl
			  << std::setw(48) << FCYN("Enter MINRES-QLP(INFO)") << std::endl
			  << "  "
			  << UNDL("                                                     ")
			  << "\n\n"
			  << std::setw(12) << "n ="        << std::setw(6) << n
			  << std::setw(12) << "||b|| ="    << std::setw(6) << beta1
			  << std::setw(12) << "precon ="   << std::setw(6) << ((precon_) ? "true" : "false")
			  << std::endl
			  << std::setw(12) << "itnlim ="   << std::setw(6) << itnlim_
			  << std::setw(12) << "rtol ="     << std::setw(6) << rtol_
			  << std::setw(12) << "shift ="    << std::setw(6) << shift_
			  << std::endl
			  << std::setw(12) << "raxxnorm =" << std::setw(6) << maxxnorm_
			  << std::setw(12) << "Acondlim =" << std::setw(6) << Acondlim_
			  << std::setw(12) << "trancond =" << std::setw(6) << trancond_
			  << std::endl
			  << "  "
			  << UNDL("                                                     ")
			  << std::endl << std::endl;
	}

	y = b, r1 = b;

	if (precon_) client.Msolve(n, &b[0], &y[0]);

	beta1 = std::inner_product(b.begin(), b.end(), y.begin(), 0.0);

	if (beta1 < 0 && _dnrm2(n, &y[0], 1) > _eps) istop_ = 11;

	if (beta1 == 0) istop_ = 3;

	beta1 = std::sqrt(beta1);

	if (checkA_ && precon_) {
		client.Msolve(n, &y[0], &r2[0]);
		s = std::inner_product(y.begin(), y.end(), y.begin(), 0.0),
		t = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0);
		double z = std::abs(s-t), epsa = std::pow((std::abs(s) + _eps)*_eps, 0.33333);
		if (z > epsa) istop_ = 10;
	}

	if (checkA_) {
		client.Aprod(n, &y[0], &w[0]);
		client.Aprod(n, &w[0], &r2[0]);
		s = std::inner_product(w.begin(), w.end(), w.begin(), 0.0),
		t = std::inner_product(y.begin(), y.end(), r2.begin(), 0.0);
		double z = std::abs(s-t), epsa = std::pow((std::abs(s) + _eps)*_eps, 0.33333);
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
			betan = _dnrm2(n, &y[0], 1);
		} else {
			client.Msolve(n, &r2[0], &y[0]);
			betan = std::inner_product(r2.begin(), r2.end(), y.begin(), 0.0);
			if (betan > 0) {
				betan = std::sqrt(betan);
			} else if (_dnrm2(n, &y[0], 1) > _eps) {
				istop_ = 11;
				break;
			}
		}

		if (itn_ == 1) {
			vec2[0] = alfa, vec2[1] = betan;
			pnorm = _dnrm2(2, &vec2[0], 1);
		} else {
			vec3[0] = beta, vec3[1] = alfa, vec3[2] = betan;
			pnorm = _dnrm2(3, &vec3[0], 1);
		}

		dbar = dltan;
		double dlta = cs*dbar + sn*alfa;
		epln = eplnn;
		double gbar = sn*dbar - cs*alfa;
		eplnn = sn*betan, dltan = -cs*betan;
		double dlta_QLP = dlta;

		gamal3 = gamal2, gamal2 = gamal, gamal  = gama;

		_symortho(gbar, betan, cs, sn, gama);
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
          		_symortho(gamal, dlta, cr1, sr1, gamal_tmp);
          		gamal     = gamal_tmp;
          		vepln     = sr1 * gama;
          		gama      = -cr1 * gama;
		}

       		// update xnorm
                xnorml = xnorm_;
       		ul4    = ul3;
       		ul3    = ul2;

		if (itn_ > 2) ul2 = ( taul2 - etal2 * ul4 - veplnl2 * ul3 ) / gamal2;
		if (itn_ > 1) ul  = ( taul  - etal  * ul3 - veplnl  * ul2) / gamal;

       		vec3[0] = xl2norm, vec3[1] = ul2, vec3[2] = ul;

       		double xnorm_tmp = _dnrm2(3, &vec3[0], 1);  // norm([xl2norm ul2 ul]);

       		if (std::abs(gama) > _eps) {
          		u = (tau - eta*ul2 - vepln*ul) / gama;
     			likeLS  = relAresl < relresl;
		        vec2[0] = xnorm_tmp, vec2[1] = u;
          		if (likeLS && _dnrm2(2, &vec2[0], 1) > maxxnorm_) {
             			u = 0;
             			istop_ = 12;
			}
		} else {
			u = 0;
			istop_ = 14;
		}

		vec2[0] = xl2norm, vec2[1] = ul2;
       		xl2norm = _dnrm2(2, &vec2[0], 1);
       		vec3[0] = xl2norm, vec3[1] = ul, vec3[2] = u;
       		xnorm_  = _dnrm2(3, &vec3[0], 1);

		// MINRES updates
        	if (Acond_ < trancond_ && istop_ == flag0 && QLPiter == 0) {
			wl2 = wl;
			wl = w;
			if (gama_tmp > _eps) {
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
       		_symortho(gamal_tmp, eplnn, cr2, sr2, gamal);
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

       		Acondl = Acond_;
       		Acond_ = Anorm_ / gmin;
       		rnorml   = rnorm_;
       		relresl = relres;

       		if (istop_ != 14) rnorm_ = phi;
       		relres = rnorm_ / (Anorm_ * xnorm_ + beta1);
       		vec2[0] = gbar, vec2[1] = dltan;
       		double rootl = _dnrm2(2, &vec2[0], 1);
		Arnorml  = rnorml * rootl;
       		relAresl = rootl / Anorm_;

       		// see if any of the stopping criteria are satisfied.
       		double epsx = Anorm_*xnorm_*_eps;

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
		} else if (betan < _eps) {
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
          		Arnorm_ = _dnrm2(n, &wl2[0], 1);
          		if (rnorm_ > 0 && Anorm_ > 0) relAres = Arnorm_ / (Anorm_*rnorm_);
		}

		if(print) _printstate(itn_-1, x1last, xnorml, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl);
		
	}

	// end of iteration loop.
        client.istop  =  istop_, client.itn   =   itn_, client.rnorm = rnorm_;
	client.Arnorm = Arnorm_, client.xnorm = xnorm_, client.Anorm = Anorm_;
        client.Acond  = Acond_;

	if (print) {
		_printstate(itn_, x[0], xnorm_, rnorm_, Arnorm_, relres, relAres, Anorm_, Acond_);
		std::cout << "  " << FCYN("Exit MINRES-QLP") << ": "
			  << msg[istop_-1] << "\n\n";
	}
}

} // end namespace minresqlp
