import numpy as np
# multi grid V cycle based off of Jim Demmel's multigrid code
def mgv(x,b,jac1,jac2):

	n = np.size(x[0])
	if (n==3):
		z = np.zeros((n,n))
		z[1,1] = b[1,1]/float(4)
		return z
	else:
		w=2./3

		for i in range(0, jac1):
			x[1:n-1, 1:n-1] = (1-w)*x[1:n-1, 1:n-1] + \
			( float(w)/4) *(x[0:n-2, 1:n-1] - x[2:n, 1:n-1] + \
							x[1:n-1, 0:n-2] - x[1:n-1, 2:n] + \
							b[1:n-1, 1:n-1])
		r = np.zeros((n,n))
		tmp =b[1:n-1, 1:n-1] -  (4 * x[1:n-1, 1:n-1] - \
			 x[0:n-2,1:n-1]  - x[2:n, 1:n-1]   - \
			 x[1:n-1, 0:n-2] - x[1:n-1, 2:n])
		r[1:n-1, 1:n-1] = tmp

		m=(n+1.)/2;
		rhat = np.zeros([m,m], dtype=np.double)
		rhat[1:m-1, 1:m-1] = .25* r[2:n-2:2, 2:n-2:2]+ \
							.125*(r[1:n-3:2, 2:n-2:2]+  r[3:n-1:2, 2:n-1:2]+ \
							      r[2:n-2:2, 1:n-3:2]+  r[2:n-2:2, 3:n-1:2])+ \
						   .0625*(r[1:n-3:2, 1:n-3:2]+  r[3:n-1:2, 1:n-3:2]+ \
						          r[1:n-3:2, 3:n-1:2]+  r[3:n-1:2 ,3:n-1:2])
		

		xhat = mgv(np.zeros((m,m), dtype=np.float), 4*rhat,jac1, jac1)


		xcor = np.zeros(np.shape(x), np.double)
		xcor[2:n-2:2, 2:n-2:2] = xhat[1:m-1, 1:m-1]
		xcor[1:n-1:2, 2:n-2:2] = .5*(xhat[0:m-1, 1:m-1]  + xhat[1:m, 1:m-1])
		xcor[2:n-2:2, 1:n-1:2] = .5*(xhat[1:m-1, 0:m-1]  + xhat[1:m-1, 1:m])
		xcor[1:n-1:2, 1:n-1:2] =.25*(xhat[0:m-1, 0:m-1]  + xhat[0:m-1, 1:m] +\
									 xhat[1:m,   0:m-1]  + xhat[1:m,   1:m])

		z = x + xcor
		for i in range(0, jac2):
			z[1:n-1, 1:n-1] = (1-w)*z[1:n-1, 1:n-1] + \
			( float(w)/4) *(z[0:n-2, 1:n-1] + z[2:n, 1:n-1] + \
							z[1:n-1, 0:n-2] + z[1:n-1, 2:n] + \
							b[1:n-1, 1:n-1])
		return z


#b=np.array([np.arange(1, 6), np.arange(6,11),np.arange(11,16),np.arange(16,21), np.arange(21,26)],dtype=np.double )
#print mgv(np.zeros_like(b,dtype=np.double), b, 1, 1 )

