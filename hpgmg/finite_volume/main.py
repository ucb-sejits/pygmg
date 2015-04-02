from __future__ import print_function, division
from hpgmg.finite_volume import VECTOR_F, VECTOR_UTRUE
from hpgmg.finite_volume.boundary_condition import BoundaryCondition
from hpgmg.finite_volume.operators import misc
from hpgmg.finite_volume.operators.problem_sine import ProblemInitializer
from hpgmg.finite_volume.space import Space

__author__ = 'nzhang-dev'

import argparse
import os
import logging

from hpgmg.finite_volume.level import Level
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius

log = logging
log.root.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log2_box_dim', help='The dimensions of the box taken log 2', default=6, type=int)
    parser.add_argument('target_boxes_per_rank', help='number of boxes per rank', type=int)
    parser.add_argument('-bc', '--boundary-conditions', dest='boundary_condition',
                        help="Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d",
                        default=('p' if os.environ.get('USE_PERIODIC_BC', 0) else 'd'),
                        choices=['p', 'd'])
    parser.add_argument('-m', '--max_coarse-dim',
                        help='Maximum coarse dim',
                        default=int(os.environ.get('MAX_COARSE_DIM', 11)),
                        type=int,
                        dest='max_coarse_dim')
    parser.add_argument('-eq', '--equation-type', help='Use h for Helmholtz or p for Poisson',
                        default=('h' if os.environ.get('USE_HELMHOLTZ', 0) else 'p'),
                        choices=['p', 'h'],
                        dest='eq')

    args = parser.parse_args()
    if args.log2_box_dim < 4:
        log.error('log2_box_dim must be at least 4')
        exit(0)

    if args.target_boxes_per_rank < 1:
        log.error('target_boxes_per_rank must be at least 1')
        exit(0)
    return args

if __name__ == '__main__':
    command_line_args = get_args()

    # basic setup
    box_dim = 1 << command_line_args.log2_box_dim
    num_tasks = 1
    target_boxes = num_tasks * command_line_args.target_boxes_per_rank
    boxes_in_i = -1
    for bi in range(1, 1001):
        total_boxes = bi**3
        if total_boxes <= target_boxes:
            coarse_grid_dim = box_dim*bi
            while coarse_grid_dim % 2 == 0:
                coarse_grid_dim /= 2
            if coarse_grid_dim <= command_line_args.max_coarse_dim:
                boxes_in_i = bi

    print("box_dim {} boxes in i-th dimension {}".format(box_dim, boxes_in_i))

    if boxes_in_i < 1:
        log.error('failed to find an acceptable problem size')
        exit(0)

    my_rank = 0
    ghosts = stencil_get_radius()
    boundary_condition = BoundaryCondition.get(command_line_args.boundary_condition)
    fine_level = Level(
        Space([boxes_in_i*box_dim for _ in range(3)]),
        Space([boxes_in_i for _ in range(3)]),
        Space([ghosts for _ in range(3)]),
        boundary_condition,
        my_rank=my_rank,
        num_ranks=8,
    )

    #conditional setup for Helmholtz and Poisson
    if command_line_args.eq == 'h':
        a = b = 1.0
        log.info('Creating Helmholtz (a={a}, b={b} test problem)'.format(a=a, b=b))
    elif command_line_args.eq == 'p':
        a = 0.0
        b = 1.0
        log.info('Creating Poisson (a={a}, b={b} test problem)'.format(a=a, b=b))
    else:
        a = b = 0.0
        print("must select either Helmoltz or Poisson")
        exit(1)

    h0 = 1 / (boxes_in_i * box_dim)
    
    ProblemInitializer.setup(fine_level, h0, a, b, is_variable_coefficient=True)

    print("Problem is setup, level.alpha_is_zero {}".format(fine_level.alpha_is_zero))

    if (a == 0.0 or fine_level.alpha_is_zero ) and fine_level.boundary_condition.is_periodic():
        # Poisson with periodic
        # nominally, u shifted by a constant is still a valid solution
        # however by convention we assume u sums to zero
        average_of_f = misc.mean(fine_level, VECTOR_UTRUE)
        if my_rank == 0:
            print("  average value of u_true = {:20.12e}... shifting u_true to ensure it sums to zero...".format(
                average_of_f))
            misc.shift_vector(fine_level, VECTOR_F, VECTOR_F, -average_of_f)

    if fine_level.boundary_condition.is_periodic():

        if average_of_f != 0.0:
            print("WARNING: Periodica bouncary conditions")

