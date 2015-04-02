from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


def dot(level, id_a, id_b):
    a_dot_b_level = 0.0

    for box in level.my_boxes:
        a_dot_b_block = 0.0
        grid_a = box.vectors[id_a]
        grid_b = box.vectors[id_b]

        for element_index in level.box_element_space.points:
            a_dot_b_block += grid_a[element_index] * grid_b[element_index]
        a_dot_b_level += a_dot_b_block

    # todo: need the body translated here

    return a_dot_b_level