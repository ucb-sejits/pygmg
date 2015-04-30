
from __future__ import print_function
import time

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np


def dot(level, id_a, id_b):
    a_dot_b_level = 0.0

    for box in level.my_boxes:
        a_dot_b_block = 0.0
        grid_a = box.vectors[id_a]
        grid_b = box.vectors[id_b]

        for element_index in level.box_element_space.points:
            a_dot_b_block += grid_a[element_index] * grid_b[element_index]
        a_dot_b_level += a_dot_b_block

    return a_dot_b_level


def mean(level, vector_id):
    start_time = time.clock()

    sum_of_level = 0.0
    elements = 0
    for box in level.my_boxes:
        sum_of_level += np.sum(box.vectors[vector_id])
        elements += level.box_element_space.volume

    level.cycles.blas1 += time.clock() - start_time

    # todo: MPI reduction stuff needs to go here

    return sum_of_level / float(elements)


def shift_vector(level, vector_id_a, vector_id_b, shift_value):
    start_time = time.clock()

    for box in level.my_boxes:
        grid_a = box.vectors[vector_id_a]
        grid_b = box.vectors[vector_id_b]

        for index in level.box_element_space.points:
            grid_b[index] = grid_a[index] + shift_value


    level.cycles.blas1 += time.clock() - start_time