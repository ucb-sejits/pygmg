from __future__ import print_function

__author__ = 'nzhang-dev'

import pymg3d
import numpy as np
import smoothers
import THeatgenerator

import timeit
import cProfile
import itertools
import sys, os


def error(A, b, x):
    return np.linalg.norm(np.dot(A, x) - b, 1)

def benchmark(A, b, x, iterations, repeats):
    smooths = (smoothers.gauss_siedel,)
    restrictions = (pymg3d.restrict, pymg3d.restrict_m, pymg3d.simple_restrict)
    interpolations = (pymg3d.interpolate, pymg3d.interpolate_m)
    timings = {}
    for smooth, restrict, interpolate in itertools.product(smooths, restrictions, interpolations):
        solver = pymg3d.MultigridSolver(interpolate, restrict, smooth, iterations//2)
        f = lambda: solver(A, b, x)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        timings[(smooth, restrict, interpolate)] = (total_time, error(A, b, result))
    return timings

def numpy_benchmark(A, b, x, iterations, repeats):
    f = lambda: np.linalg.solve(A, b)
    total_time = timeit.Timer(f).timeit(repeats)
    result = f()
    return {(np.linalg.solve, None, None): (total_time, error(A, b, result))}

def smoother_benchmark(A, b, x, iterations, repeats):
    smooths = (smoothers.gauss_siedel,)
    timings ={}
    for smooth in smooths:
        f = lambda: smooth(A, b, x, iterations)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        timings[(smooth, None, None)] = (total_time, error(A, b, result))
    return timings

if __name__ == '__main__':
    length = int(sys.argv[1])
    iterations = int(sys.argv[2])
    repeats = eval(sys.argv[3])
    ndim = 2
    shape = (length,) * ndim
    seed = sum(ord(i) for i in 'SEJITS')
    np.random.seed(seed)
    z = np.random.ranf()*32
    d1 = np.random.ranf()*z
    d2 = np.random.ranf()*(z-d1)
    A = np.identity(shape[0])*(z)
    A += np.diag((d1,)*(shape[0]-1), 1)
    A += np.diag((d2,)*(shape[0]-1), -1)
    b = np.random.random((shape[0],))
    b = np.random.random((length,)*ndim)
    x = np.zeros_like(b)
    results = {}
    for bench in (benchmark, numpy_benchmark, smoother_benchmark):
        results.update(bench(A, b, x, iterations, repeats))
    print("Smoother".ljust(20), "Interpolate".ljust(20), "Restrict".ljust(20), "Time/run".ljust(20), "Error".ljust(20), sep='\t')
    print("-"*120)
    for key, value in results.items():
        for k in key:
            print((k.__name__ if k is not None else 'None').ljust(20), end="\t")
        print(str(value[0]/iterations).ljust(20), str(value[1]))


