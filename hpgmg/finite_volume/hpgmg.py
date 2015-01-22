__author__ = 'nzhang-dev'

import argparse
from __future__ import print_function

MAX_COARSE_DIM = 11

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log2_box_dim', help='The dimensions of the box taken log 2', type=int)
    parser.add_argument('target_boxes_per_rank', 'number of boxes per rank', type=int)

    args = parser.parse_args()
    if args.log2_box_dim < 4:
        print('log2_box_dim must be at least 4')
        exit(0)

    if args.target_boxes_per_rank < 1:
        print('target_boxes_per_rank must be at least 1')
        exit(0)
    return args.log2_box_dim, args.target_boxes_per_rank

if __name__ == '__main__':
    log2_box_dim, target_boxes_per_rank = get_args()
    box_dim = 1 << log2_box_dim
    num_tasks = 1
    target_boxes = num_tasks * target_boxes_per_rank
    boxes_in_i = -1
    for bi in range(1, 1001):
        total_boxes = bi**3
        if total_boxes <= target_boxes:
            coarse_grid_dim = box_dim*bi
            while coarse_grid_dim % 2 == 0:
                coarse_grid_dim /= 2
            if coarse_grid_dim <= MAX_COARSE_DIM:
                boxes_in_i = bi

    if boxes_in_i < 1:
        print('failed to find an acceptable problem size')
        exit(0)