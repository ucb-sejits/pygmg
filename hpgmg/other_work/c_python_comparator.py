from __future__ import print_function
import sys
import csv
from collections import defaultdict
import numpy as np
import numpy.testing as np_test

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Comparator(object):
    def __init__(self, allowed_percent_delta):
        self.grid_names = []
        self.kind_to_grids = defaultdict(dict)
        self.allowed_percent_delta = allowed_percent_delta

    def read_file(self, name, kind):
        with open(name, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line, row in enumerate(reader):
                try:
                    if not row or row[0] != "==":
                        pass
                    elif row[1] == "MESHSTART":
                        grid_name = row[2]
                        shape = (int(row[3]), int(row[4]), int(row[5]))
                        print("mesh kind {} start {} shape {}".format(kind, grid_name, shape))
                        if kind == 'c':
                            self.grid_names.append(grid_name)
                        self.kind_to_grids[kind][grid_name] = np.empty(shape)
                        in_mesh = True
                    elif row[1] == "MESHEND":
                        print("mesh end")
                        in_mesh = False
                        # print(self.kind_to_grids[kind][grid_name])
                    elif row[1] == "block":
                        pass
                    elif row[0] == "==":
                        if in_mesh:
                            # print("data {} {}".format(row[1], row[2]))
                            i = int(row[1])
                            j = int(row[2])
                            for k in range(shape[2]):
                                self.kind_to_grids[kind][grid_name][(i, j, k)] = float(row[k+3])
                except Exception as e:
                    print("err line {} '{}'".format(line, row))
                    raise e


    def percent_delta(self, desired, actual):
        if actual != 0.0:
            return abs(desired-actual)/float(abs(actual))
        elif desired != 0.0:
            return abs(desired-actual)/float(abs(desired))
        return 0.0

    def values_close_enough(self, desired, actual):
        # print("des {} act {} del {} allow {} ret {}".format(
        #     desired, actual, self.percent_delta(desired, actual), self.allowed_percent_delta,
        #     self.percent_delta(desired, actual) <= self.allowed_percent_delta
        # ))
        return self.percent_delta(desired, actual) <= self.allowed_percent_delta

    def show_difference(self, c_grid, py_grid):
        shown = 0
        for i in range(c_grid.shape[0]):
            for j in range(c_grid.shape[1]):
                for k in range(c_grid.shape[2]):
                    if not self.values_close_enough(c_grid[(i, j, k)], py_grid[(i, j, k)]):
                        print("diff {:15s} c {:15.6e} py {:15.6e} %-err {:15.6f} allowed {}".format(
                            (i, j, k),
                            c_grid[(i, j, k)], py_grid[(i, j, k)],
                            self.percent_delta(c_grid[(i, j, k)], py_grid[(i, j, k)]),
                            self.allowed_percent_delta))
                        shown += 1
                        if shown > 160:
                            return

    def compare_grids(self):
        for grid_name in self.grid_names:
            print("comparing grid {}".format(grid_name))
            c_grid = self.kind_to_grids['c'][grid_name]
            py_grid = self.kind_to_grids['py'][grid_name]
            try:
                np_test.assert_array_almost_equal(c_grid, py_grid, allowed_percent_delta=6)
            except Exception:
                self.show_difference(c_grid, py_grid)

if __name__ == '__main__':
    allowed_percent_delta = .0001 if len(sys.argv) < 4 else float(sys.argv[3])

    c = Comparator(allowed_percent_delta=allowed_percent_delta)

    c.read_file(sys.argv[1], 'c')
    c.read_file(sys.argv[2], 'py')

    print("keys {}".format(c.kind_to_grids.keys()))
    c.compare_grids()