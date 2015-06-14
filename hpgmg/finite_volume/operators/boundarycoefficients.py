
import numpy as np

from sympy import *
from sympy.abc import x,y

class BoundaryCoefficients:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = np.array(degree)
        if degree == 2:
            self.coefficients = self.generateV2()
        elif degree ==4:
            self.coefficients = self.generateV4()


    def generateV2(self):
        a, b, c = symbols('a, b, c')
        l = []
        # append n simulaneous equations coefficients l[i] into list
        for i in range(0,self.degree):
            f = Poly(a*x**2 + b*x , x)
            l.append(integrate(f, (x, i, i+1)).coeffs())
        A = np.array(l)
        B =  np.linalg.inv(A) # "solve" system
        uneg1 = integrate(f, (x, -1, 0)).coeffs() #
        coeffs = np.zeros(self.degree)
        for i in range(self.degree):
            coeffs = coeffs + B[i]*uneg1[i]
        return coeffs

    def generateV4(self): #NOT DONE YET
        a, b, c, d = symbols('a, b, c, d')
        l = []
        # append n simulaneous equations coefficients l[i] into list
        for i in range(0,self.degree):
            f = Poly(a*x**4 + b*x**2 + c*x + d, x)
            l.append(integrate(f, (x, i, i+1)).coeffs())
        A = np.array(l)
        B =  np.linalg.inv(A) # "solve" system
        print B


    #a coefficient grid is generated directly the self.coefficients
    # boundary type is either "corner", "edge", or "face"
    def generateV2CoefficientGrid(self, boundaryType):
        x = self.coefficients
        if boundaryType == "corner":
            coefficient_grid = np.ndarray([2,2,2])
            for i in range(0, self.degree):
                for j in range(0, self.degree):
                    for k in range(0, self.degree):
                        coefficient_grid[i][j][k] = x[i]*x[j]*x[k]
            return coefficient_grid

        elif boundaryType == "edge":
            coefficient_grid = np.ndarray([2,2])
            for i in range(0, self.degree):
                for j in range(0, self.degree):
                    coefficient_grid[i][j] = x[i]*x[j]
            return coefficient_grid

        elif boundaryType == "face":
                return self.coefficients

        else:
            print("invalid boundary type")



if __name__== "__main__":
	BC = BoundaryCoefficients(2)
	print "corner"
	print BC.generateV2CoefficientGrid("corner")
	print "edge"
	print BC.generateV2CoefficientGrid("edge")
	print "face "
	print BC.generateV2CoefficientGrid("face")

