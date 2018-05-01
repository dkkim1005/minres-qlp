#include <iostream>
#include <vector>
#include <assert.h>
#include <complex>
#include "minresqlp.h"

typedef std::complex<double> dcomplex;

// BLAS definitions
extern "C" {
	void dgemv_(const char *TRANS, const int *M, const int *N,
                    const double *ALPHA, const double *A, const int *LDA,
                    const double *X, const int *INCX, const double *BETA,
                          double *Y, const int *INCY);
	void zgemv_(const char *TRANS, const int *M, const int *N, const dcomplex *ALPHA,
		    const dcomplex *A, const int *LDA, const dcomplex *X, const int *INCX,
		    const dcomplex *BETA, dcomplex *Y, const int *INCY);
}

void
dgemv(const int m, const int n, const double alpha, const double* a, const double *x, const double beta, double *y)
{
        const char trans = 'T';
        const int inc = 1, lda = n;

        dgemv_(&trans, &n, &m, &alpha, a, &lda, x, &inc, &beta, y, &inc);
}

void
zgemv(const int m, const int n, const dcomplex alpha, const dcomplex *a, const dcomplex *x,
      const dcomplex beta, dcomplex *y)
{
	const char trans = 'T';
	const int inc = 1, lda = n;

	zgemv_(&trans, &n, &m, &alpha, a, &lda, x, &inc, &beta, y, &inc);
}


// format for a real type matrix 
struct real_type : public minresqlp::baseInfo<double> {
	real_type(const int n_, const std::vector<double>& b_, const std::vector<double>& A_)
	: baseInfo<double>(n_, b_), _A(A_), _nrowA(A_.size()/b_.size()) {
 		assert(b_.size() == n_);
	}

	virtual void
	Aprod(const int n, const double *x, double *y) const override{
		dgemv(_nrowA, n, 1, &_A[0], x, 0, y);
	}

private:
	const std::vector<double> _A;
	const int _nrowA;
};



// format for a hermitian matrix
struct hermite_type : public minresqlp::baseInfo<dcomplex> {
	hermite_type(const int n_, const std::vector<dcomplex>& b_, const std::vector<dcomplex>& A_)
	: baseInfo<dcomplex>(n_, b_), _A(A_), _nrowA(A_.size()/b_.size()) {}

	virtual void
	Aprod(const int n, const dcomplex *x, dcomplex *y) const {
		zgemv(_nrowA, n, 1, &_A[0], x, 0, y);
	}

private:
	const std::vector<dcomplex> _A;
	const int _nrowA;
};


int main(int argc, char* argv[])
{
	const int N = 20;
	std::vector<double> b(N), A(N*N);

	for(int i=0; i<N; ++i) {
		b[i] = i+1;
		for(int j=0; j<N; ++j) {
			A[i*N + j] = i*j;
		}
	}

	real_type client(N, b, A);

	minresqlp::realSolver<real_type> solver;

	solver.solve(client, true);

	std::cout << "  vector x: "
		  << std::setprecision(5);
	if(N < 10) {
		for(const auto &xi :client.x) std::cout << std::setw(7) << xi << " ";
	} else {
		for(int i=0; i<5; ++i)   std::cout << std::setw(7) << client.x[i] << " ";
		std::cout << ".... ";
		for(int i=N-5; i<N; ++i) std::cout << std::setw(7) << client.x[i] << " ";
	}
	std::cout << "\n\n";

	std::vector<dcomplex> zb(N), zA(N*N);

	for(int i=0; i<N; ++i) {
		zb[i] = dcomplex(i,i+3);
		zA[i*N+i] = dcomplex(i, 0)/dcomplex(N,0);
		for(int j=i+1; j<N; ++j) {
			zA[i*N + j] = dcomplex(5*i, j)/dcomplex(N,0);
			zA[j*N + i] = std::conj(zA[i*N+j]);
		}
	}

	hermite_type zclient(N, zb, zA);
	zclient.itnlim = 20*zb.size();
	zclient.shift  = 1e-3;

	minresqlp::hermitianSolver<hermite_type> zsolver;

	zsolver.solve(zclient, true);

	std::cout << "  vector x: "
		  << std::setprecision(5);
	if(N < 10) {
		for(const auto &xi :client.x) std::cout << std::setw(7) << xi << " ";
	} else {
		for(int i=0; i<5; ++i)   std::cout << std::setw(7) << client.x[i] << " ";
		std::cout << ".... ";
		for(int i=N-5; i<N; ++i) std::cout << std::setw(7) << client.x[i] << " ";
	}
	std::cout << "\n\n";

	return 0;
}
