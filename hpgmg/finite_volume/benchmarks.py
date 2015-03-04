from __future__ import print_function, division

__author__ = 'nzhang-dev'

import pymg3d
import numpy as np
import smoothers
import THeatgenerator

import timeit
import itertools
import sys
import argparse
import os
import logging



log = logging
log.root.setLevel(logging.INFO)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('length', help='The length of the box taken log 2(length-1)', default=6, type=int)

    parser.add_argument('iterations', help='iterations', default=6, type=int)

    parser.add_argument('ndim', help='number of dimensions of grid', default=6, type=int)

    parser.add_argument('repeats', help='repeats', default=6, type=str)

    if False: #FIX ME !!!!!!
        parser.add_argument('-m', '--max_coarse-dim',
                            help='Maximum coarse dim',
                            #default=int(os.environ.get('MAX_COARSE_DIM', 11)),
                            type=int,
                            dest='max_coarse_dim')

    parser.add_argument('--DUSE_CHEBY', dest='cheby', action='store_true')
    parser.add_argument('--DUSE_GS', dest='gs', action='store_true')
    parser.add_argument('--DUSE_JACOBI', dest='j', action='store_true')


    args = parser.parse_args()


    smoother_xor= int(args.cheby) + int(args.gs) + int(args.j) 
    if(smoother_xor != 1 ):
        log.error('Must choose one smoother (CHEBY, GS, or JACOBI')
        exit(0)
    return args

def error(A, b, x):
    return np.linalg.norm(np.dot(A, x.flatten()) - b.flatten(), 1)

def benchmark(A, b, x, iterations, repeats):
    #smooths = (smoothers.gauss_siedel,)
    smooths = (smoother_choice,)
    restrictions = (pymg3d.restrict, pymg3d.restrict_m, pymg3d.simple_restrict)
    interpolations = (pymg3d.interpolate, pymg3d.interpolate_m)
    timings = []
    for smooth, restrict, interpolate in itertools.product(smooths, restrictions, interpolations):
        solver = pymg3d.MultigridSolver(interpolate, restrict, smooth, iterations//2)
        f = lambda: solver(A, b, x)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        yield (smooth.__name__, restrict.__name__, interpolate.__name__, total_time, error(A, b, result))
    

def numpy_benchmark(A, b, x, iterations, repeats):
    results = []
    for solver in (np.linalg.solve,):
        f = lambda: solver(A, b.flatten())
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        yield (solver.__name__, None, None, total_time, error(A, b, result))
    

def smoother_benchmark(A, b, x, iterations, repeats):
    smooths = (smoothers.gauss_siedel,)
    for smooth in smooths:
        f = lambda: smooth(A, b, x, iterations)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        yield (smooth.__name__, None, None, total_time, error(A, b, result))

if __name__ == '__main__':
    
    
    global smoother_choice
    smoother_choice = smoothers.gauss_siedel


    args = get_args()
    smoother_enum = int(args.cheby) + 2*int(args.gs) + 3*int(args.j)


    global smoother_choice
    smoother_options={
                1:smoothers.chebyshev, 
                2:smoothers.gauss_siedel,
                3:smoothers.jacobi
    }
    smoother_choice = smoother_options[smoother_enum]


    length = args.length
    iterations = args.iterations
    ndim = args.ndim
    repeats = eval(args.repeats)

    print()
    print("Length:", length, "iterations:", iterations, "ndim:", ndim, "repeats:", repeats, sep="\t")
    delta = 1
    C = 0.2
    h = 1/length
    shape = (length,) * ndim
    A = THeatgenerator.heat_matrix(length, C, delta, h, ndim)

    b = np.random.random((shape[0],))
    b = np.random.random((length,)*ndim)
    x = np.zeros_like(b)
    results = []
    print("Starting Benchmarks")
    print("Smoother".ljust(20), "Interpolate".ljust(20), "Restrict".ljust(20), "Time/run".ljust(20), "Error".ljust(20), sep='\t')
    print("-"*120)
    for bench in (benchmark, numpy_benchmark, smoother_benchmark):

        results = bench(A, b, x, iterations, repeats)
        for result in results:
            print(*[str(i).ljust(20) for i in result], sep="\t")




