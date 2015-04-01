from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Constants(object):
    # ------------------------
    VECTOR_TEMP = 0  #
    VECTOR_UTRUE = 1  # exact solution used to generate f
    VECTOR_F_MINUS_AV = 2  # cell centered residual (f-Av)
    # ------------------------
    VECTOR_F = 3  # original right-hand side (Au=f), cell centered
    VECTOR_U = 4  # numerical solution
    VECTOR_ALPHA = 5  # cell centered coefficient
    VECTOR_BETA_I = 6  # face centered coefficient (n.b. element = 0 is the left face of the ghost zone element)
    VECTOR_BETA_J = 7  # face centered coefficient (n.b. element = 0 is the back face of the ghost zone element)
    VECTOR_BETA_K = 8  # face centered coefficient (n.b. element = 0 is the bottom face of the ghost zone element)
    # ------------------------
    VECTOR_DINV = 9  # cell centered relaxation parameter (e.g. inverse of the diagonal)
    VECTOR_L1INV = 10  # cell centered relaxation parameter (e.g. inverse of the L1 norm of each row)
    VECTOR_VALID = 11  # cell centered array noting which cells are actually present
    # ------------------------
    VECTORS_RESERVED = 12  # total number of grids and the starting location for any auxillary bottom solver grids
    # ------------------------

    @classmethod
    def vector_list(cls):
        return [
            cls.VECTOR_TEMP,
            cls.VECTOR_UTRUE,
            cls.VECTOR_F_MINUS_AV,
            cls.VECTOR_U,
            cls.VECTOR_ALPHA,
            cls.VECTOR_BETA_I,
            cls.VECTOR_BETA_J,
            cls.VECTOR_BETA_K,
            cls.VECTOR_DINV,
            cls.VECTOR_L1INV,
            cls.VECTOR_VALID,
        ]
