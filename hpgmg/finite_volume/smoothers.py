import numpy as np
import scipy as sp
import functools

def flatten_args(func):
    """
    :param func: function to wrap, flattens all of the args to 2-D, 1D, 1D
    :return: wrapped function
    """
    @functools.wraps(func)
    def f(A, b, x, iter=10):
        b_flat = b.flatten()
        x_flat = x.flatten()
        return func(A, b_flat, x_flat, iter).reshape(x.shape)
    return f

# Gauss Siedel solver.
@flatten_args
def gauss_siedel(A, b, x, iter = 10):
    L = np.tril(A)  # lower triangular matrix of A, includes diagonal
    Linv = np.linalg.inv(L)  # inverse of L
    U = A-L  # strictly upper triangular matrix of A
    for _ in range(iter):
        x = (Linv.dot(b-(U.dot(x))))     #iterate
        #print x
    return x

# Gauss Siedel which requires only one storage array
# From Wikipedia page on Gauss-Siedel smoothing
@flatten_args
def gauss_siedel_inplace(A, b, x, iter = 10):
    for _ in range(iter):
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
    return x
 
 #saad's algorithm for chebychev: does not work
def chebyshevbad(A,b, x, diag, diaginv, iterations=9, delta=1, theta=1):
    r = b - np.dot(A,x)
    sigma = float(delta)/theta
    rho = 1./sigma
    d = (1./theta)*r
    for _ in range(iterations):
        x = x + d
        r = r - np.dot(A, d)
        rho1 = 1./(2*sigma - rho)
        d = rho*rho1*d - (2.*rho1/delta)*r
        rho = rho1
    return x


# Chebychev acceleration 
def chebyshev(A, b1, x1, diag, diaginv, iterations, c,d):
    '''
    This method uses chebyshev acceleration to smooth solution to Ax=b

    :param A: The operator matrix in the Ax=b equation, as a numpy array
    :param b1: The solution matrix in the Ax=b equation, as a numpy array
    :param x1: The input matrix in the Ax=b equation, as a numpy array
    :param diag: A with all but diagonal entries stripped away
    :param diaginv: inverse of diag
    :param iterations: The number of iterations the smoothing is to be applied for
    :param c: half-width of range of spectrum of A
    :param d: average of range of spectrum of A
    :return: a smoothed version of the input matrix x, the same shape as it was inputted in as.
    '''
    x=x1.flatten()
    b=b1.flatten()

    r = b - np.dot(A,x)
    for i in range(0,iterations):
        z = np.dot(diaginv,r)
        if i==0:
            rho = z
            alpha = 2. / d
        else:
            beta = (c * alpha / (2.)) * (c * alpha / (2.))
            alpha = 1. / (d - beta)
            rho = z + beta * rho
        x = x + alpha * rho
        r = b - np.dot(A,x)
    return x.reshape(x1.shape)

" Uses Gershgorin circle theorem to approximate spectrum of A"
def apprxspectrum(A):
    a=2


def get_smoother(type):
    smoother = globals().get
    return smoother(type)

def smooth_matrix(A, b, x, iterations = 10, smooth_func=gauss_siedel):
    '''
    This method is the wrapper for accessing smooth functions in this file. It handles the flatting
    of input matrices prior to smoothing, calling the appropriate smoothing function, and reconstituting
    the flattened, smoothed matrix back to its original shape.

    :param A: The operator matrix in the Ax=b equation, as a numpy array
    :param b: The solution matrix in the Ax=b equation, as a numpy array
    :param x: The input matrix in the Ax=b equation, as a numpy array
    :param iterations: The number of iterations the smoothing is to be applied for
    :param smooth_func: The function required for smoothing. Default is the gauss_siedel function
    :return: a smoothed version of the input matrix x, the same shape as it was inputted in as.
    '''
    input_shape_x = x.shape
    x_flat = x.flatten()

    smooth_x = smooth_func(A, b, x_flat, iterations)

    expanded_smooth_x = smooth_x.reshape(input_shape_x)
    return expanded_smooth_x


if __name__=="__main__":
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8.]])

    print "A:"
    print A 
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    diag = np.diag(np.diag(A))
    dinv=np.linalg.inv(diag)
    eigenvalues=np.linalg.eig(diag)

    alpha= min(eigenvalues[0])
    beta = max(eigenvalues[0])
    c = float(beta-alpha)/2.
    d = float(beta+alpha)/2.
    r=get_smoother("gauss_siedel")
    print "chebychev", chebyshev(A, b , x ,diag, dinv, 100, c,d)


    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    print "gs", r(A, b, x, 10)
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    #print "gs in place ", gauss_siedel_inplace(A, b, x, 10)


