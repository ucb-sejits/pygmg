from __future__ import print_function
from collections import Iterable

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class PythonBlock(object):
    def __init__(self, parent=None, indent=0):
        if parent is None:
            self.parent = self
            self.lines = []
        else:
            self.parent = parent
        self.indent = indent

    def __enter__(self):
        return self

    def __iadd__(self, other):
        self.parent.lines.append("{}{}".format(" "*self.indent, other))

    def append(self, other):
        self.parent.lines.append("{}{}".format(" "*self.indent, other))

    def __exit__(self, *args):
        pass

    def block(self):
        return PythonBlock(self.parent, self.indent+2)

    def __call__(self, *args, **kwargs):
        self.lines += args[0]

if __name__ == '__main__':
    with PythonBlock() as b0:
        with b0.block() as b:
            for x in range(5):
                b.append("x += {}".format(x))
            with b.block() as bb:
                for x in range(5):
                    bb.append("y += {}".format(x))
            for x in range(5):
                b.append("z += {}".format(x))

    print("\n".join(b0.lines))


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
