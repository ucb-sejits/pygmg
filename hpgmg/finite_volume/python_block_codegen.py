from __future__ import print_function

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
