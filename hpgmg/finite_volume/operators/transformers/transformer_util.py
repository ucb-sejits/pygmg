import functools

__author__ = 'nzhang-dev'


def get_obj(namespace, name):
    parts = name.split('.')
    obj_name = parts.pop(0)
    try:
        obj = namespace[obj_name]
    except KeyError:
        raise NameError("{} is not defined in namespace {}".format(obj_name, namespace))
    for part in parts:
        obj = getattr(obj, part)
    return obj

def nest_loops(loops):
    top, bottom = loops[0], loops[-1]
    fold_op = lambda x, y: (x.body.append(y), y)[1]
    functools.reduce(fold_op, loops)
    return top, bottom
