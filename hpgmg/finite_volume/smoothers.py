import numpy as np

#gauss siedel solver. 
def gausssiedel(A, b, x, iter=10):
    L = np.tril(A)  # lower triangular matrix of A, includes diagonal
    Linv = np.linalg.inv(L)  # inverse of L
    U = A-L  # strictly upper triangular matrix of A
    for _ in range(iter):
        x = (Linv.dot(b-(U.dot(x))))     #iterate
        #print x
    return x

#gauss siedel which requires only one storage array
#from wikipedia page on Gauss-Siedel smoothing
def gausssiedel_inplace(A, b, x, iter=10):
    for _ in range(iter):
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        #print x
    return x
 

#chebyshev smoother THIS DOES NOT WORK YET
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


def get_smoother(type):
    smoother = globals().get
    return smoother("gausssiedel")


if __name__=="__main__":
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8.]])
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    print gausssiedel(A, b, x, 10)
    b = np.array([6., 25., -11., 15.])
    x = np.zeros_like(b)
    print "in place "
    print gausssiedel_inplace(A, b, x, 10)