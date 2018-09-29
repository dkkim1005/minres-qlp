import numpy as np
from __minresqlp import symmetric_solver, hermitian_solver

def minresqlp(A, b, **kwargs):
    """
    	* default setting for the parameters
                shift = 0.0;
                M = [];
                disable = false;
                itnlim = -1;
                rtol = 1e-16;
                maxxnorm= 1e7;
                trancond = 1e7;
                Acondlim = 1e15;
                print = false;
    """

    A = np.array(A); b = np.array(b)

    shapeA = A.shape
    shapeb = b.shape

    # dimension check
    if shapeA != (shapeb[0], shapeb[0]) or \
       len(shapeA) != 2 or len(shapeb) != 1:
       raise Exception(" Check your dimension of inputs")

    dtypeA = A.dtype
    dtypeb = b.dtype

    dset = set([dtypeA, dtypeb])

    # type test
    if   np.dtype('complex64')  in dset : dtype = 'complex'
    elif np.dtype('complex128') in dset : dtype = 'complex'
    elif np.dtype('float32')    in dset : dtype = 'float'
    elif np.dtype('float64')    in dset : dtype = 'float'
    elif np.dtype('int64')      in dset : dtype = 'float'
    else: raise Exception(" Check your type of inputs")

    if dtype == 'complex':
        A = A.astype('complex128'); b = b.astype('complex128')
        M = kwargs.get('M')
        if M is not None:
            M = np.array(M).astype('complex128')
            if M.shape != A.shape: raise Exception(" Check your dimension of inputs")
            kwargs['useMsolve'] = True
        else: M = np.array([]).astype('complex128')

        return hermitian_solver(b, A, kwargs, M)

    elif dtype == 'float':
        A = A.astype('float64'); b = b.astype('float64')
        M = kwargs.get('M')
        if M is not None:
            M = np.array(M).astype('float64')
            if M.shape != A.shape: raise Exception(" Check your dimension of inputs")
            kwargs['useMsolve'] = True
        else: M = np.array([]).astype('float64')

        return symmetric_solver(b, A, kwargs, M)


if __name__ == "__main__":
    A = [[1, 10, 3], [10, 10, 4], [3, 4, 10]]
    b = [1,2,3]
    print minresqlp(A, b, printinfo = True, itnlim = 100, rtol = 1e-10, disable = False)
    print minresqlp(A, b, printinfo = True, itnlim = 100, rtol = 1e-10, disable = False, M = np.eye(len(A)))
    print minresqlp(A, b, printinfo = True, itnlim = 100, rtol = 1e-10, disable = False, M = np.diag([1./(A[i][i]+1e-5) for i in range(len(A))]))
    
