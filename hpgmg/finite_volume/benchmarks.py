from __future__ import print_function, division

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
    return np.linalg.norm(np.dot(A, x.flatten()) - b.flatten(), 1)

def benchmark(A, b, x, iterations, repeats):
    smooths = (smoothers.gauss_siedel,)
    restrictions = (pymg3d.restrict, pymg3d.restrict_m, pymg3d.simple_restrict)
    interpolations = (pymg3d.interpolate, pymg3d.interpolate_m)
    timings = []
    for smooth, restrict, interpolate in itertools.product(smooths, restrictions, interpolations):
        solver = pymg3d.MultigridSolver(interpolate, restrict, smooth, iterations//2)
        f = lambda: solver(A, b, x)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        timings.append((smooth.__name__, restrict.__name__, interpolate.__name__, total_time, error(A, b, result)))
    return timings

def numpy_benchmark(A, b, x, iterations, repeats):
    results = []
    for solver in (np.linalg.solve,):
        f = lambda: solver(A, b.flatten())
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        results.append((solver.__name__, None, None, total_time, error(A, b, result)))
    return results

def smoother_benchmark(A, b, x, iterations, repeats):
    smooths = (smoothers.gauss_siedel,)
    timings = []
    for smooth in smooths:
        f = lambda: smooth(A, b, x, iterations)
        total_time = timeit.Timer(f).timeit(repeats)
        result = f()
        timings.append((smooth.__name__, None, None, total_time, error(A, b, result)))
    return timings

if __name__ == '__main__':
    length = int(sys.argv[1])
    iterations = int(sys.argv[2])
    repeats = eval(sys.argv[3])
    ndim = 3
    delta = 1
    C = 0.2
    h = 1/length
    shape = (length,) * ndim
    A = THeatgenerator.gen3DHeatMatrixT(length, C, delta, h)

    b = np.random.random((shape[0],))
    b = np.random.random((length,)*ndim)
    x = np.zeros_like(b)
    results = []
    for bench in (numpy_benchmark, benchmark, smoother_benchmark):
        results.extend(bench(A, b, x, iterations, repeats))
    print("Smoother".ljust(20), "Interpolate".ljust(20), "Restrict".ljust(20), "Time/run".ljust(20), "Error".ljust(20), sep='\t')
    print("-"*120)
    for result in results:
        print(*[str(i).ljust(20) for i in result], sep="\t")


