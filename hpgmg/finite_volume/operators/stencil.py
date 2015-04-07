from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg.finite_volume.constants import Constants
from hpgmg.finite_volume.operators.restriction import Restriction, restriction


class Stencil(object):
    def rebuild_operator(self, target_level, source_level, a, b):
        if target_level.my_rank == 0:
            print("  rebuilding 27pt CC operator for level...  h=%e  ".format(target_level.h))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # form restriction of alpha[], beta_*[] coefficients from fromLevel
        if source_level is None:
            restriction(target_level, Constants.VECTOR_ALPHA,
                        source_level, Constants.VECTOR_ALPHA, Restriction.RESTRICT_CELL  );
            restriction(target_level, Constants.VECTOR_BETA_I,
                        source_level, Constants.VECTOR_BETA_I, Restriction.RESTRICT_FACE_I);
            restriction(target_level, Constants.VECTOR_BETA_J,
                        source_level, Constants.VECTOR_BETA_J, Restriction.RESTRICT_FACE_J);
            restriction(target_level, Constants.VECTOR_BETA_K,
                        source_level, Constants.VECTOR_BETA_K, Restriction.RESTRICT_FACE_K);
        # else case assumes alpha/beta have been set

