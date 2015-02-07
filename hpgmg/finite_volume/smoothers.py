import numpy as np

# Gauss Siedel solver.
def gauss_siedel(A, b, x, iter = 10):
    print(A, b, x, iter)
    L = np.tril(A)  # lower triangular matrix of A, includes diagonal
    Linv = np.linalg.inv(L)  # inverse of L
    U = A-L  # strictly upper triangular matrix of A
    for _ in range(iter):
        x = (Linv.dot(b-(U.dot(x))))     #iterate
        #print x
    return x

# Gauss Siedel which requires only one storage array
# From Wikipedia page on Gauss-Siedel smoothing
def gauss_siedel_inplace(A, b, x, iter = 10):
    for _ in range(iter):
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
    return x
 

#chebyshev smoother THIS DOES NOT WORK YET
# Chebyshev smoother THIS DOES NOT WORK YET
def chebyshev(A, b, x, iter=10):
    
    for k in range(iter):
        r=b-A.dot(x)
        if  k==0:
            alpha=1/d
        elif k==1:
            alpha=2*d*(1/(2*d*d-c*c))
        elif k>1:
            alpha=1/((d-alpha*c*c)/4)
    beta=alpha*d-1
    p=alpha*r+beta*p
    x=x+p

    # FIXME: This method is non-functional at the moment
    return None

def get_smoother(type):
    smoother = globals().get
    return smoother("gauss_siedel")


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
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    print gauss_siedel(A, b, x, 10)
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    print "in place "
    print gauss_siedel_inplace(A, b, x, 10)
