from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot(mesh, shape=None, title='surface', use_heat=True, ghost_size=1):
    """
    plots mesh, removing ghost zone if specified.

    :param mesh: a 2d shape, currently this
    :param shape:
    :param title:
    :param use_heat:
    :return:
    """
    surface = np.array(mesh[ghost_size:-ghost_size, ghost_size:-ghost_size])
    surface = surface if not shape else surface.reshape(shape)
    xs, ys = np.indices(surface.shape)

    if not use_heat:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xs, ys, surface)
        plt.title("{}, average is {}".format(title, np.average(surface)))
    else:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.set_title("{}, average is {:8.4}".format(title, np.average(surface)))
        plt.imshow(surface)  # , vmin=-30, vmax=30)
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        plt.show()
