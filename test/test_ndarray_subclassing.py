from __future__ import print_function
import unittest
import numpy as np

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class InfoArray(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.my_new_attribute = 0
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            self.my_new_attribute =22
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.my_new_attribute = getattr(obj, 'my_new_attribute', 0)
        # We do not need to return anything


class TestNDArraySubclassing(unittest.TestCase):
    """
    The Mesh class used in PyGMG is a first class citizen in the project and is a
    subclass of the numpy ndarray.  Subclassing ndarray has quite a bit of lore
    associated with it.  This test is largelu
    """

    def test_basics(self):
        one_d_obj = InfoArray((5, ))
        one_d_obj[1] = 1
        one_d_obj[2] = 2
        one_d_obj[2] = 43
        self.assertTrue(hasattr(one_d_obj, "my_new_attribute"))

        two_d_obj = InfoArray((5, 5))
        two_d_obj[1][1] = 1
        two_d_obj[2][2] = 2
        two_d_obj[2][3] = 43
        self.assertTrue(hasattr(one_d_obj, "my_new_attribute"))

        two_d_obj = InfoArray((5, 5))
        two_d_obj.my_new_attribute += 10
        two_d_obj[(1, 1)] = 1
        two_d_obj.my_new_attribute += 10
        two_d_obj[(2,2)] = 2
        two_d_obj.my_new_attribute += 10
        two_d_obj[2, 2] = 43
        self.assertEqual(two_d_obj.my_new_attribute, 30)