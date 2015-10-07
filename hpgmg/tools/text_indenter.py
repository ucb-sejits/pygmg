from __future__ import print_function
from collections import Iterable

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TextIndenter(object):
    def __init__(self):
        self.lines = []
        self.indent_amount = 0

    def do_indent(self, string):
        self.lines.append("{}{}".format(" "*self.indent_amount, string))

    def __iadd__(self, other):
        if isinstance(other, basestring):
            self.do_indent(other)
        elif isinstance(other, Iterable):
            for line in other:
                self.do_indent(line)
        return self

    def indent(self):
        self.indent_amount += 4

    def outdent(self):
        self.indent_amount -= 4

    def __str__(self):
        return "\n".join(self.lines)
