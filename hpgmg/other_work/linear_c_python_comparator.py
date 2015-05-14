from __future__ import print_function
import sys
import csv
import math
from collections import defaultdict
import numpy as np
import numpy.testing as np_test

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Comparator(object):
    def __init__(self, allowed_percent_delta):
        self.c_grids, self.c_grid_names, self.c_start_lines = [], [], []
        self.py_grids, self.py_grid_names, self.py_start_lines = [], [], []

        self.kind_to_grids = defaultdict(dict)
        self.allowed_percent_delta = allowed_percent_delta

    def read_file(self, name, kind):
        if kind == 'c':
            grids, grid_names, start_lines = self.c_grids, self.c_grid_names, self.c_start_lines
        else:
            grids, grid_names, start_lines = self.py_grids, self.py_grid_names, self.py_start_lines

        rows = 0
        shape = None
        grid_name = ''
        current_grid = None
        in_mesh = False

        with open(name, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line, row in enumerate(reader):
                try:
                    if not row:
                        pass
                    elif row[0] != "==":
                        pass
                    elif row[1] == "MESHSTART":
                        rows = 0
                        grid_name = row[2]
                        shape = (int(row[3]), int(row[4]), int(row[5]))
                        print("mesh kind {} start {} shape {}".format(kind, grid_name, shape))
                        grid_names.append(grid_name)
                        start_lines.append(line+1)
                        current_grid = np.empty(shape)
                        grids.append(current_grid)
                        in_mesh = True
                    elif row[1] == "MESHEND":
                        print("mesh end rows {}".format(rows))
                        in_mesh = False
                        # print(self.kind_to_grids[kind][grid_name])
                    elif row[1] == "block":
                        pass
                    elif row[0] == "==":
                        if in_mesh:
                            rows += 1
                            # print("data {} {}".format(row[1], row[2]))
                            i = int(row[1])
                            j = int(row[2])
                            for k in range(shape[2]):
                                current_grid[(i, j, k)] = float(row[k+3])
                except Exception as e:
                    print("err line {} '{}'".format(line, row))
                    raise e

    @staticmethod
    def percent_delta(desired, actual):
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
        return Comparator.percent_delta(desired, actual) <= self.allowed_percent_delta

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
        allowed_difference = abs(int(math.log(self.allowed_percent_delta, 10)))
        print("allowed difference {}".format(allowed_difference))

        for index, c_grid in enumerate(self.c_grids):
            c_grid_name = self.c_grid_names[index]
            py_grid_name = self.py_grid_names[index]

            if c_grid_name != py_grid_name:
                print("Bad match at {} c grid {} vs py grid {}".format(index, c_grid_name, py_grid_name))
                break

            print("comparing {} starting line c {} py {}".format(
                c_grid_name, self.c_start_lines[index], self.py_start_lines[index]
            ))

            py_grid = self.py_grids[index]

            try:
                np_test.assert_array_almost_equal(c_grid, py_grid, decimal=allowed_difference)
            except Exception:
                self.show_difference(c_grid, py_grid)
            print("compare done")

if __name__ == '__main__':
    threshold = .0001 if len(sys.argv) < 4 else float(sys.argv[3])

    c = Comparator(allowed_percent_delta=threshold)

    c.read_file(sys.argv[1], 'c')
    c.read_file(sys.argv[2], 'py')

    # print("keys {}".format(c.grid_names))
    c.compare_grids()