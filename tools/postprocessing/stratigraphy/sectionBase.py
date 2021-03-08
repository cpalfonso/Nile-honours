import os

import matplotlib.pyplot as plt
import numpy as np

from . import crossSection
from ...grid import Grid


class _SectionBase:
    def __init__(self, a, b, input_topo=None, order='ij'):
        if order == 'xy':
            a = a[::-1]
            b = b[::-1]
        elif order != 'ij':
            raise ValueError(
                "Unrecognised 'order' parameter: {}.".format(order)
                + "\nValid values are 'ij' and 'xy'."
            )
        self.a = a
        self.b = b
        self.input_topo = input_topo

    @classmethod
    def _from_xy(cls, a, b, input_topo, order='xy'):
        if order == 'yx':
            a = a[::-1]
            b = b[::-1]
        elif order != 'xy':
            raise ValueError(
                "Unrecognised 'order' parameter: {}.".format(order)
                + "\nValid values are 'xy' and 'yx'."
            )
        grid = Grid(input_topo)
        x_min = grid.x.min()
        y_min = grid.y.min()
        dx = grid.x.flatten()[1] - grid.x.flatten()[0]
        a = (
            int((a[0] - x_min) / dx),
            int((a[1] - y_min) / dx),
        )
        b = (
            int((b[0] - x_min) / dx),
            int((b[1] - y_min) / dx),
        )
        section = cls(a, b, input_topo, order='xy')
        return section

    def plot_strike(self, ax=None, **kwargs):
        '''
        Plot the strike of the cross-section.

        ----------------------------------------------------------------------
        Parameters (optional):
            ax: if given, plot will be drawn on this matplotlib.axes.Axes
                object.
                (default None)
            **kwargs: any further kwargs are passed to
                matplotlib.pyplot.plot or matplotlib.axes.Axes.plot.
        '''
        x = (self.a[1], self.b[1])
        y = (self.a[0], self.b[0])
        if ax is None:
            plt.plot(x, y, **kwargs)
        else:
            ax.plot(x, y, **kwargs)

    def _section_points(self, num_points):
        x = np.linspace(self.a[1], self.b[1], num_points)
        y = np.linspace(self.a[0], self.b[0], num_points)
        points = np.vstack((
            y.reshape((1, -1)),
            x.reshape((1, -1)),
        ))
        return points

    def _project_endpoints(self):
        if self.input_topo is None:
            raise AttributeError("input_topo has not been set.")
        grid = Grid(self.input_topo)
        a_proj = grid.index_to_coords(*self.a)
        b_proj = grid.index_to_coords(*self.b)
        return a_proj, b_proj

    def _section_dists(self, num_points):
        a_proj, b_proj = self._project_endpoints()
        dist = np.sqrt(
            (a_proj[1] - b_proj[1]) ** 2
            + (a_proj[0] - b_proj[0]) ** 2
        )
        dists = np.linspace(0, dist, num_points)
        return dists

    def get_dists(self, num_points, units='km'):
        dists = self._section_dists(num_points)
        if units == 'km':
            dists *= 1.e-3
        return dists

    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, tup):
        if len(tup) != 2:
            raise ValueError("A must be of length 2.")
        self._a = tup

    @property
    def b(self):
        return self._b
    @b.setter
    def b(self, tup):
        if len(tup) != 2:
            raise ValueError("B must be of length 2.")
        self._b = tup

    @property
    def input_topo(self):
        return self._input_topo
    @input_topo.setter
    def input_topo(self, filename):
        if filename is not None:
            if not os.path.isfile(filename):
                raise FileNotFoundError(
                    "Topography file not found: '{}'".format(filename)
                )
        self._input_topo = filename

    def _to_CrossSection(self, sed_file=None):
        cs = crossSection.CrossSection(self.a, self.b, self.input_topo, sed_file=sed_file)
        return cs
