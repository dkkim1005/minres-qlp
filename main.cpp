#include <iostream>
#include <vector>
#include <assert.h>
#include "minresqlp.h"

extern "C" void dgemv_(const char *TRANS, const int *M, const int *N,
                       const double *ALPHA, const double *A, const int *LDA,
                       const double *X, const int *INCX, const double *BETA,
                       double *Y, const int *INCY);

inline void dgemv(const int m, const int n, const double alpha, const double* a, const double *x, const double beta, double *y) {
        const char trans = 'T';
        const int inc = 1, lda = n;

        dgemv_(&trans, &n, &m, &alpha, a, &lda, x, &inc, &beta, y, &inc);
}

struct Info : public minresqlp::baseInfo {
	Info(const int n_, const std::vector<double>& b_, const std::vector<double>& A_)
	: baseInfo(n_, b_), _A(A_), _nrowA(A_.size()/b_.size()) {
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


int main(int argc, char* argv[])
{
	std::vector<double> b = {3, 4, 3, 1, 2, 3},

			    A = { 2, 1, 3, 0, 0, 3,
				  1, 1, 2, 0, 0, 0,
				  3, 2, 1, 0, 0, 0,
				  0, 0, 0, 1, 1, 1,
				  0, 0, 0, 1, 2, 1,
                                  3, 0, 0, 1, 1, 1};

	Info client(6, b, A);

	minresqlp::mainsolver<Info> solver;

	solver.solve(client, true);

	std::cout << "  vector x:"
		  << std::setprecision(5);
	for(const auto &xi :client.x) std::cout << std::setw(7) << xi << " ";
	std::cout << "\n\n";

	return 0;
}
