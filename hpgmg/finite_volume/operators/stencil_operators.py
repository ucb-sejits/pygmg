import numpy as np
from abc import ABCMeta, abstractmethod

class Operator:
  __metaclass__ = ABCMeta
  @abstractmethod
  def apply_op(self, i, j, k):
    pass

class ConstantCoefficent7pt(Operator):
  def __init__(self, a,b, h2inv):
    self.a = a
    self.b = b
    self.h2inv = h2inv

  def apply_op(self, x, i, j, k):
    jStride = len(x)**(1.0/3)
    kStride = jStride**2
    ijk = i + j*jStride + k*kStride          
    return self.a*x[ijk] - self.b*self.h2inv*(    \
      + x[ijk+1      ]             \
      + x[ijk-1      ]             \
      + x[ijk+jStride]             \
      + x[ijk-jStride]             \
      + x[ijk+kStride]             \
      + x[ijk-kStride]             \
      - x[ijk        ]*6.0         \
    )     

if __name__=="__main__":

  xf = np.linspace(0.0, 63.0, num=64)
  A = ConstantCoefficent7pt(1,2, .1)
  print(A.apply_op(xf, 1,1,1))

