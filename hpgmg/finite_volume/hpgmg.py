from __future__ import print_function, division

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
    parser.add_argument('-bc', '--boundary-conditions',
                        help='Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d or USE_PERIODIC_BC from environment',
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
    args = get_args()

    #basic setup
    box_dim = 1 << args.log2_box_dim
    num_tasks = 1
    target_boxes = num_tasks * args.target_boxes_per_rank
    boxes_in_i = -1
    for bi in range(1, 1001):
        total_boxes = bi**3
        if total_boxes <= target_boxes:
            coarse_grid_dim = box_dim*bi
            while coarse_grid_dim % 2 == 0:
                coarse_grid_dim /= 2
            if coarse_grid_dim <= args.max_coarse_dim:
                boxes_in_i = bi

    if boxes_in_i < 1:
        log.error('failed to find an acceptable problem size')
        exit(0)

    ghosts = stencil_get_radius()
    calc_level = Level(boxes_in_i, box_dim, ghosts,)

    #conditional setup for Helmholtz and Poisson
    if args.eq == 'h':
        a = b = 1
        log.info('Creating Helmholtz (a={a}, b={b} test problem)'.format(a=a, b=b))
    elif args.eq == 'p':
        a = 0
        b = 1
        log.info('Creating Poisson (a={a}, b={b} test problem)'.format(a=a, b=b))

    h0 = 1 / (boxes_in_i * box_dim)



