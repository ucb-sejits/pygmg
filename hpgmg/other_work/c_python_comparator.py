from __future__ import print_function
import sys
import csv
from collections import defaultdict
import numpy as np
import numpy.testing as np_test

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Comparator(object):
    def __init__(self, precision):
        self.kind_to_grids = defaultdict(dict)
        self.decimal = precision

    def read_file(self, name, kind):
        with open(name, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if not row or row[0] != "==":
                    pass
                elif row[1] == "MESHSTART":
                    grid_name = row[2]
                    shape = (int(row[3]), int(row[4]), int(row[5]))
                    print("mesh kind {} start {} shape {}".format(kind, grid_name, shape))
                    self.kind_to_grids[kind][grid_name] = np.empty(shape)
                    in_mesh = True
                elif row[1] == "MESHEND":
                    print("mesh end")
                    in_mesh = False
                    # print(self.kind_to_grids[kind][grid_name])
                elif row[1] == "block":
                    pass
                elif row[0] == "==":
                    # print("data {} {}".format(row[1], row[2]))
                    i = int(row[1])
                    j = int(row[2])
                    for k in range(shape[2]):
                        self.kind_to_grids[kind][grid_name][(i, j, k)] = float(row[k+3])

    def compare(self, desired, actual):
        return abs(desired-actual) < 0.5 * 10**(-self.decimal)

    def compare_grids(self):
        for grid_name in self.kind_to_grids['c']:
            print("comparing grid {}".format(grid_name))
            c_grid = self.kind_to_grids['c'][grid_name]
            py_grid = self.kind_to_grids['py'][grid_name]
            try:
                np_test.assert_array_almost_equal(c_grid, py_grid, decimal=6)
            except Exception:
                shown = 0
                for i in range(c_grid.shape[0]):
                    for j in range(c_grid.shape[1]):
                        for k in range(c_grid.shape[2]):
                            if not self.compare(c_grid[(i, j, k)], py_grid[(i, j, k)]):
                                print("diff {:15s} c {:15.6e} py {:15.6e} delta {}".format(
                                    (i, j, k),
                                    c_grid[(i, j, k)], py_grid[(i, j, k)],
                                    c_grid[(i, j, k)] - py_grid[(i, j, k)]))
                                shown += 1
                                if shown > 16:
                                    return

if __name__ == '__main__':
    precision = 6 if len(sys.argv) < 4 else int(sys.argv[3])

    c = Comparator(precision=precision)

    c.read_file(sys.argv[1], 'c')
    c.read_file(sys.argv[2], 'py')

    print("keys {}".format(c.kind_to_grids.keys()))
    c.compare_grids()