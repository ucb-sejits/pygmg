from snowflake.stencil_compiler import CCompiler
from snowflake_openmp.compiler import TiledOpenMPCompiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

CONFIG = None

compiler = None

def setup():
    global compiler
    if CONFIG.backend == 'c':
        compiler = CCompiler()
    if CONFIG.backend == 'omp':
        compiler = TiledOpenMPCompiler()
