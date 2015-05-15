from __future__ import print_function
import sys
import csv
import math
from collections import defaultdict
import numpy as np
import numpy.testing as np_test

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class GridReader(object):
    def __init__(self, grid_file_name):
        self.grid_file_name = grid_file_name
        self.grid_file = open(grid_file_name, 'rb')
        self.reader = csv.reader(self.grid_file, delimiter=',')
        self.last_start_line = 0
        self.rows = 0
        self.shape = None
        self.current_grid = None
        self.current_grid_name = ''
        self.in_mesh = False
        self.verbose = False

    def next_row(self):
        for line, row in enumerate(self.reader):
            yield line, row

    def next_grid(self):
        while True:
            row = self.reader.next()
            # print("row is {}".format(row))
            if row is None:
                return None
            line = self.reader.line_num
            try:
                if not row:
                    pass
                elif row[0] != "==":
                    pass
                elif row[1] == "MESHSTART":
                    rows = 0
                    self.current_grid_name = row[2]
                    shape = (int(row[3]), int(row[4]), int(row[5]))
                    self.last_start_line = line
                    if self.verbose:
                        print("{} mesh start line {} shape {}".format(self.current_grid_name, line, shape))
                    self.current_grid = np.empty(shape)
                    self.in_mesh = True
                elif row[1] == "MESHEND":
                    if self.verbose:
                        print("mesh end rows {}".format(rows))
                    self.in_mesh = False
                    return self.current_grid
                elif row[1] == "block":
                    pass
                elif row[0] == "==":
                    if self.in_mesh:
                        rows += 1
                        # print("data {} {}".format(row[1], row[2]))
                        i = int(row[1])
                        j = int(row[2])
                        for k in range(shape[2]):
                            self.current_grid[(i, j, k)] = float(row[k+3])
            except Exception as e:
                print("err line {} '{}'".format(line, row))
                raise e


class Comparator(object):
    def __init__(self, file_name_1, file_name_2, allowed_percent_delta):
        self.file_names = [file_name_1, file_name_2]
        self.grid_ids = [x for x in range(len(self.file_names))]
        self.allowed_percent_delta = allowed_percent_delta
        self.grids_compared = 0

        self.grid_readers = [
            GridReader(self.file_names[grid_id])
            for grid_id in self.grid_ids
        ]

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

    def is_different(self, c_grid, py_grid):
        for i in range(c_grid.shape[0]):
            for j in range(c_grid.shape[1]):
                for k in range(c_grid.shape[2]):
                    if not self.values_close_enough(c_grid[(i, j, k)], py_grid[(i, j, k)]):
                        return True
        return False

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
        while True:
            grid_0 = self.grid_readers[0].next_grid()
            if grid_0 is None:
                return
            grid_1 = self.grid_readers[1].next_grid()
            if grid_1 is None:
                print("Second file did not contain matching mesh named {} starting at line {}".format(
                    self.grid_readers[0].current_grid_name, self.grid_readers[0].last_start_line
                ))
                return

            self.grids_compared += 1
            grid_name_0 = self.grid_readers[0].current_grid_name
            grid_name_1 = self.grid_readers[1].current_grid_name
            start_line_0 = self.grid_readers[0].last_start_line
            start_line_1 = self.grid_readers[1].last_start_line
            print("comparing {}:{}:{} to {}:{}:{}".format(
                self.file_names[0], grid_name_0, start_line_0,
                self.file_names[1], grid_name_1, start_line_1,
            ))
            if grid_name_0 != grid_name_1:
                print("ERROR: These grids do not match")
                break

            if self.is_different(grid_0, grid_1):
                self.show_difference(grid_0, grid_1)
            print("compare done")

if __name__ == '__main__':
    threshold = .0001 if len(sys.argv) < 4 else float(sys.argv[3])

    c = Comparator(sys.argv[1], sys.argv[2], allowed_percent_delta=threshold)

    c.compare_grids()

    print("compared {} grids".format(c.grids_compared))