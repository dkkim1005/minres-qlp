#include <boost/python.hpp>
#include "../minresqlp.h"
#include "numpy_array.h"

typedef std::complex<double> dcomplex;

extern "C" {
void dsymv_(const char *UPLO, const int *N, const double *ALPHA, const double *A,
		    const int *LDA, const double *X, const int *INCX,
		    const double *BETA, double *Y, const int *INCY);

void zhemv_(const char *UPLO, const int *N, const dcomplex *ALPHA, const dcomplex *A,
		    const int *LDA, const dcomplex *X, const int *INCX,
		    const dcomplex *BETA, dcomplex *Y, const int *INCY);
}

namespace blas
{
	void
	dsymv(const int n, const double alpha, const double *a,
	      const double *x, const double beta, double *y)
	{
		const char uplo = 'u';
		const int inc = 1;
		dsymv_(&uplo, &n, &alpha, a, &n, x, &inc, &beta, y, &inc);
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
}

typedef boost::python::dict BoostPyDict;

namespace py_minresqlp_wrapper
{
	struct parameterset
	{
		// default options
                double shift_     = 0.0;
                bool   useMsolve_ = false;
                bool   disable_   = false;
                int    itnlim_    = -1;
                double rtol_      = 1e-16;
                double maxxnorm_  = 1e7;
                double trancond_  = 1e7;
                double Acondlim_  = 1e15;
                bool   print_     = false;

		void get_options(const BoostPyDict& info)
		{
			if (info.has_key("shift"))     shift_     = boost::python::extract<double>(info.get("shift"));
			if (info.has_key("useMsolve")) useMsolve_ = boost::python::extract<bool>(info.get("useMsolve"));
			if (info.has_key("disable"))   disable_   = boost::python::extract<bool>(info.get("disable"));
			if (info.has_key("itnlim"))    itnlim_    = boost::python::extract<int>(info.get("itnlim"));
			if (info.has_key("rtol"))      rtol_      = boost::python::extract<double>(info.get("rtol"));
			if (info.has_key("maxxnorm"))  maxxnorm_  = boost::python::extract<double>(info.get("maxxnorm"));
			if (info.has_key("trancond"))  trancond_  = boost::python::extract<double>(info.get("trancond"));
			if (info.has_key("Acondlim"))  Acondlim_  = boost::python::extract<double>(info.get("Acondlim"));
			if (info.has_key("printinfo")) print_     = boost::python::extract<bool>(info.get("printinfo"));
		}
	};

	struct symmetric_type : public minresqlp::baseInfo<double>
	{
		symmetric_type(const int n, const std::vector<double>& b, const std::vector<double>& A, const parameterset& para, const double *M = NULL)
		: baseInfo<double>(n, b, para.shift_, para.useMsolve_,
				   para.disable_, para.itnlim_, para.rtol_,
				   para.maxxnorm_, para.trancond_, para.Acondlim_,
				   para.print_), _A(A) {
			blas::transpose(n, &_A[0]);
			if (M != NULL) _M.assign(M, M + n*n);
		}

		virtual void
		Aprod(const int n, const double *x, double *y) const {
			blas::dsymv(n, 1, &_A[0], &x[0], 0, &y[0]);
		}

		virtual void
		Msolve(const int n, const double *x, double *y) const {
			if(useMsolve) blas::dsymv(n, 1, &_M[0], &x[0], 0, &y[0]);
		}
	
	private:
		std::vector<double> _A, _M;
	};

	struct hermitian_type : public minresqlp::baseInfo<dcomplex>
	{
		hermitian_type(const int n, const std::vector<dcomplex>& b, const std::vector<dcomplex>& A, const parameterset& para, const dcomplex *M = NULL)
		: baseInfo<dcomplex>(n, b, para.shift_, para.useMsolve_,
				     para.disable_, para.itnlim_, para.rtol_,
				     para.maxxnorm_, para.trancond_, para.Acondlim_,
				     para.print_), _A(A)
		{
			blas::transpose(n, &_A[0]);
			if (M != NULL) _M.assign(M, M + n*n);
		}

		virtual void
		Aprod(const int n, const dcomplex *x, dcomplex *y) const {
			blas::zhemv(n, 1, &_A[0], &x[0], 0, &y[0]);
		}

		virtual void
		Msolve(const int n, const dcomplex *x, dcomplex *y) const {
			if(useMsolve) blas::zhemv(n, 1, &_M[0], &x[0], 0, &y[0]);
		}

	private:
		std::vector<dcomplex> _A, _M;
	};


	PyObject* py_minresqlp_symmetric(PyObject *numpy_b, PyObject *numpy_A, BoostPyDict info, PyObject *numpy_M)
	{
		parameterset parameters;
		parameters.get_options(info);

		ndarray_to_C_ptr_wrapper<double> wrapperb(numpy_b), wrapperA(numpy_A), wrapperM(numpy_M);
		const int n = wrapperA.get_size(0), m = wrapperM.get_size(0);
		std::vector<double> b(wrapperb.get_ptr(), wrapperb.get_ptr() + n),
				    A(wrapperA.get_ptr(), wrapperA.get_ptr() + n*n);

		symmetric_type *symmetric_mode;
		if (m == 0) {
			symmetric_mode = new symmetric_type(n, b, A, parameters);
		} else {
			symmetric_mode = new symmetric_type(n, b, A, parameters, &wrapperM[0]);
		}

		minresqlp::realSolver<symmetric_type> solver;

		solver.solve(*symmetric_mode);

		npy_intp dims[1] = {symmetric_mode -> n};

		PyObject* x = C_ptr_to_ndarray_wrapper(&(symmetric_mode -> x)[0], 1, dims, NPY_FLOAT64);

		delete symmetric_mode;

		return x;
	}

	PyObject* py_minresqlp_hermitian(PyObject *numpy_b, PyObject *numpy_A, BoostPyDict info, PyObject *numpy_M)
	{
		parameterset parameters;
		parameters.get_options(info);

		ndarray_to_C_ptr_wrapper<dcomplex> wrapperb(numpy_b), wrapperA(numpy_A), wrapperM(numpy_M);
		const int n = wrapperA.get_size(0), m = wrapperM.get_size(0);
		std::vector<dcomplex> b(wrapperb.get_ptr(), wrapperb.get_ptr() + n),
				      A(wrapperA.get_ptr(), wrapperA.get_ptr() + n*n);

		hermitian_type *hermitian_mode;

		if (m == 0) {
			hermitian_mode = new hermitian_type(n, b, A, parameters);
		} else {
			hermitian_mode = new hermitian_type(n, b, A, parameters, &wrapperM[0]);
		}

		minresqlp::hermitianSolver<hermitian_type> zsolver;

		zsolver.solve(*hermitian_mode);

		npy_intp dims[1] = {hermitian_mode -> n};

		PyObject* x = C_ptr_to_ndarray_wrapper(&(hermitian_mode -> x)[0], 1, dims, NPY_COMPLEX128);

		delete hermitian_mode;

		return x;
	}
}


BOOST_PYTHON_MODULE(__minresqlp)
{
	import_array();
	boost::python::def("symmetric_solver", &py_minresqlp_wrapper::py_minresqlp_symmetric);
	boost::python::def("hermitian_solver", &py_minresqlp_wrapper::py_minresqlp_hermitian);
}
