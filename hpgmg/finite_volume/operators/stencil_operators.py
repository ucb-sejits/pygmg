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

  def apply_op1d(self, x, i, j, k):
    jStride = int(round(pow(len(x),1.0/3))) #dimension of grid with ghost regions
    kStride = jStride**2
    ijk = (1+i) + (1+j)*jStride + (1+k)*kStride #map i,j,k to ghost grid ijk
    return self.a*x[ijk] - self.b*self.h2inv*(    \
      + x[ijk+1      ]             \
      + x[ijk-1      ]             \
      + x[ijk+jStride]             \
      + x[ijk-jStride]             \
      + x[ijk+kStride]             \
      + x[ijk-kStride]             \
      - x[ijk        ]*6.0         \
    )     

  def apply_op(self, x, i,j,k):
     return self.a*x[i][j][k] - self.b*self.h2inv*(    \
      + x[i+1][j][k]         \
      + x[i-1][j][k]         \
      + x[i][j+1][k]         \
      + x[i][j-1][k]         \
      + x[i][j][k+1]         \
      + x[i][j][k-1]         \
      - x[i][j][k]*6.0         \
    )     

  def D_inv1d(self, x, i, j, k):
    #FIX ME. should simply use mesh methods to retrieve number of neighbors
    jStride = int(round(pow(len(x),1.0/3))) #dimension of grid with ghost regions
    kStride = jStride**2
    ijk = (1+i) + (1+j)*jStride + (1+k)*kStride #map i,j,k to ghost grid ijk
    return 1.0 / (self.a - self.b*self.h2inv*(            \
             + valid[ijk-1      ]   \
             + valid[ijk-jStride]   \
             + valid[ijk-kStride]   \
             + valid[ijk+1      ]   \
             + valid[ijk+jStride]   \
             + valid[ijk+kStride]   \
             - 12.0                 \
          ))      


  def D_inv(self, x, i, j, k):
    #FIX ME. should simply use mesh methods to retrieve number of neighbors
    i,j,k = i+1, j+1, k+1
    return 1.0 / (self.a - self.b*self.h2inv*(            \
             + valid[i][j][k-1]   \
             + valid[i][j][k+1]   \
             + valid[i][j+1][k]   \
             + valid[i][j-1][k]   \
             + valid[i+1][j][k]   \
              + valid[i-1][j][k]   \
             - 12.0                 \
          ))                        \


def initialize_valid_region1d(x):
  dim = int(round(pow(len(x),1.0/3)))
  jStride = dim
  kStride = dim**2
  valid = np.zeros(len(x))
 
  for ind in range(0, len(x)):
    valid[ind] = 0

  for i in range(1, dim-1):
    for j in range(1, dim-1):
      for k in range(1, dim-1):
        ijk = i + j*jStride + k*kStride   
        valid[ijk] = 1
  return valid

def getSideLength(x):
  assert x.shape[0]==x.shape[1] and x.shape[1]==x.shape[2], "Grid must be cubic"
  return x.shape[0]

def initialize_valid_region(x):
  dim = getSideLength(x)
  valid_shape = (dim+1, dim+1, dim+1) #dimensions of new valid cube
  valid = np.zeros(valid_shape)
 
  for i in range(0, dim): #initialize all cells to 0/invalide
    for j in range(0, dim):
      for k in range(0, dim): 
        valid[i][j][k] = 0

  for i in range(1, dim-1): #initialize non-ghost zone cells to 1, assuming ghost zone has width of 1
    for j in range(1, dim-1):
      for k in range(1, dim-1):
        valid[i][j][k] = 1
  return valid

def add_boundary1d(x):
  old_dim = int(round(pow(len(x),1.0/3)))
  jStride = old_dim
  kStride = old_dim**2
  
  new_dim = old_dim+2
  new_jStride = new_dim
  new_kStride = new_dim**2

  new_x = np.zeros(new_dim**3)
  for i in range(0, old_dim):
    for j in range(0, old_dim):
      for k in range(0, old_dim):
        old_ijk = i + j*jStride + k*kStride 
        new_ijk =  (1+i) + (1+j)*new_jStride + (1+k)*new_kStride  
        new_x[new_ijk] = x[old_ijk]
  return new_x

def add_boundary(x):
  old_dim = getSideLength(x)
  new_dim = old_dim+2
  new_x = np.zeros((new_dim, new_dim, new_dim)) #dimensions of new cube
  for i in range(0, old_dim):
    for j in range(0, old_dim):
      for k in range(0, old_dim):
        new_i, new_j, new_k = 1+i, 1+j, 1+k #map coordinate of old cube to new cube
        new_x[new_i][new_j][new_k] = x[i][j][k]
  return new_x

if __name__=="__main__":

  #xf = np.linspace(0.0, 63.0, num=64)
  #A = ConstantCoefficent7pt(1,2, .1)
  #print A.apply_op1d(xf, 1,1,1)

  #print x
  xf = np.linspace(0.0, 63.0, 64)
  v = initialize_valid_region1d(xf)
  print v

  xf = np.linspace(0.0, 7.0, 8)
  print add_boundary1d(xf)


