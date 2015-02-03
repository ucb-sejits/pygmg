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

#chebyshev smoother
def chebyshev(A, b, x, iter=10):
    return x


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




