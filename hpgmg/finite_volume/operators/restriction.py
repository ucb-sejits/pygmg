from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

class Restriction(object):
    RESTRICT_CELL = 0
    RESTRICT_FACE_I = 1
    RESTRICT_FACE_I = 2
    RESTRICT_FACE_I = 3
    RESTRICT_FACE_I = 4

def restriction_pc_block(level_c, vector_id_c, level_f, vector_id_f, block, restriction_type):
    if restriction_type == Restriction.RESTRICT_CELL:
        for box in level_c.my_boxes:
            for index in box.interior_points():
                box.vectors[vector_id_c][index] = (

                )


def restriction(target_level, target_vector_id, source_level, source_vector_id, restriction_type):
