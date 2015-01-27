from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BoundaryCondition(object):
    """
    encapsulate boundary condition for a level
    allocated_blocks, num_blocks and blocks array indices have the following meanings
    0 holds values for all blocks
    1 holds values for just faces

     / 24 25 26 /
    / 21 22 23 /	(k+1)
   / 18 19 20 /

     / 15 16 17 /
    / 12 13 14 /	(k)
   /  9 10 11 /

     /  6  7  8 /
    /  3  4  5 /	(k-1)
   /  0  1  2 /

    """
    All = 0
    JustFaces = 1

    PERIODIC = 0
    DIRICHLET = 1

    Faces = [
        0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0,
    ]
    Edges = [
        0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0,
    ]
    Corners = [
        1, 0, 1, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 1, 0, 1,
    ]

    Normal = 13

    @staticmethod
    def neighbor_vector(di, dj, dk):
        return BoundaryCondition.Normal + (di * 9) + (dj * 3) + dk

    @staticmethod
    def is_face(neighbor_vector):
        return BoundaryCondition.Faces[neighbor_vector] > 0

    @staticmethod
    def is_edge(neighbor_vector):
        return BoundaryCondition.Edges[neighbor_vector] > 0

    @staticmethod
    def is_corner(neighbor_vector):
        return BoundaryCondition.Corners[neighbor_vector] > 0

    @staticmethod
    def valid_index(index):
        return BoundaryCondition.All <= index <= BoundaryCondition.JustFaces

    @staticmethod
    def foreach_neighbor_delta():
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    yield di, dj, dk

    def __init__(self, condition_type):
        assert condition_type in [BoundaryCondition.DIRICHLET,BoundaryCondition.PERIODIC]

        self.condition_type = condition_type
        self.allocated_blocks = [0, 0]
        self.num_blocks = [0, 0]
        self.blocks = [[], []]
