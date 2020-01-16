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

	void zhemv_(const char *UPLO, const int *N, const dcomplex *ALPHA, const dcomplex *A,
		    const int *LDA, const dcomplex *X, const int *INCX,
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

void
zhemv(const int n, const dcomplex alpha, const dcomplex *a,
      const dcomplex *x, const dcomplex beta, dcomplex *y)
{
	const char uplo = 'u';
	const int inc = 1;
	zhemv_(&uplo, &n, &alpha, a, &n, x, &inc, &beta, y, &inc);
}

template<typename float_t>
void
transpose(const int n, float_t *A)
{
	for (int i=0; i<n; ++i) {
		for (int j=i+1; j<n; ++j) {
			const float_t temp = A[i*n + j];
			A[i*n + j] = A[j*n + i], A[j*n + i] = temp;
		}
	}
}

// format for a real type matrix 
struct real_type : public MINRESQLP::BaseInfo<double> {
	real_type(const int n_, const std::vector<double>& b_, const std::vector<double>& A_)
	: BaseInfo<double>(n_, b_), _A(A_), _nrowA(A_.size()/b_.size()) {
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
struct hermite_type : public MINRESQLP::BaseInfo<dcomplex> {
	hermite_type(const int n_, const std::vector<dcomplex>& b_, const std::vector<dcomplex>& A_)
	: BaseInfo<dcomplex>(n_, b_), _A(A_), _nrowA(A_.size()/b_.size())
	{
		transpose(n_, &_A[0]);
	}

	virtual void
	Aprod(const int n, const dcomplex *x, dcomplex *y) const {
		//zgemv(_nrowA, n, 1, &_A[0], x, 0, y);
		zhemv(n, 1, &_A[0], &x[0], 0, &y[0]);
	}

	/*
	virtual void
	Msolve(const int n, const dcomplex *x, dcomplex *y) const {
		for (int i=0; i<n; ++i) y[i] = dcomplex(1e-10, 0)*x[i];
	}
	*/

private:
	std::vector<dcomplex> _A;
	const int _nrowA;
};


int main(int argc, char* argv[])
{
	const int N = 20;
	std::vector<double> b(N), A(N*N);

	for(int i=0; i<N; ++i) {
		b[i] = i+1;
		for(int j=0; j<N; ++j) {
			A[i*N + j] = i*j + 1;
		}
	}

	real_type client(N, b, A);
	client.print = true;

	MINRESQLP::RealSolver<real_type> solver;

	std::cout << "\n\n   / Real type matrix solver(Ax=b) /" << std::endl;
	solver.solve(client);

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

	for(int i=0; i<5; ++i) std::cout << "------------------------";
	
	std::cout << std::endl;

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
	// Setting options
	zclient.useMsolve = false;
	zclient.maxxnorm  = 1e7;
	zclient.print     = true;

	std::cout << "\n\n   / Hermitian matrix solver(Ax=b) /" << std::endl;
	MINRESQLP::HermitianSolver<hermite_type> zsolver;

	zsolver.solve(zclient);

	std::cout << "  vector x: "
		  << std::setprecision(5);
	if(N < 10) {
		for(const auto &xi :zclient.x) std::cout << std::setw(7) << xi << " ";
	} else {
		for(int i=0; i<5; ++i)   std::cout << std::setw(7) << zclient.x[i] << " ";
		std::cout << ".... ";
		for(int i=N-5; i<N; ++i) std::cout << std::setw(7) << zclient.x[i] << " ";
	}
	std::cout << "\n\n";

	return 0;
}
