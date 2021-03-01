'''
Class to contain all data and metadata relating to
inputs and outputs on the Badlands regular grid.
'''
## TODO: Add docstrings for Grid properties.
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree

from .gridbuilder import GridBuilder
from . import grid_tools
from ..io.xml import get_sealevel


class Grid:
    '''
    Container for bringing together TIN, flow, and
    stratigraphy data on the one grid.

    --------------------------------------------------------------------------
    '''
    def __init__(
            self,
            input_topo,
            tin_file=None,
            flow_file=None,
            sed_file=None,
    ):
        '''
        Create a Grid object from an initial topography .csv file.

        ----------------------------------------------------------------------
        Parameters:
        --------------------------------------------------------------------------
            input_topo (str): the Badlands input topography .csv
                file (generally topo0Ma.csv).
            tin_file (str, optional): a TIN HDF5 file to map to
                the regular grid.
            flow_file (str, optional): a flow HDF5 file to map to
                the regular grid.
            sed_file (str, optional): a stratigraphy HDF5 file to map
                to the regular grid.
        '''
        try:
            if not input_topo.endswith('.csv'):
                input_topo += '.csv'
        except AttributeError as e:
            raise TypeError(
                "Invalid filename: {}".format(input_topo)
            ) from e
        rXY, nx, ny, rZ, dx = grid_tools.readRegularGrid(
            input_topo,
        )
        self._x = rXY[:, 0].reshape((ny, nx))
        self._y = rXY[:, 1].reshape((ny, nx))
        self._z = rZ
        self._nx = nx
        self._ny = ny
        self._dx = dx
        self._input_topo = input_topo

        self._tin = None
        self._sealevel = 0.0
        self._flow = None
        self._strat = None
        self.tin = tin_file
        self.flow = flow_file
        self.strat = sed_file

    def copy(self):
        if self.tin is not None:
            tin_file = self.tin.filename
        else:
            tin_file = None
        if self.flow is not None:
            flow_file = self.flow.filename
        else:
            flow_file = None
        if self.strat is not None:
            sed_file = self.strat.filename
        else:
            sed_file = None
        copy = Grid(self.input_topo, tin_file, flow_file, sed_file)
        return copy

    def get_rXY(self):
        rXY = np.hstack((
            self.x.reshape((-1, 1)),
            self.y.reshape((-1, 1)),
        ))
        return rXY

    def get_llc(self, order='xy'):
        if order in ['xy', 'ji', 'lon_lat']:
            llc = (self.x.min(), self.y.min())
        elif order in ['yx', 'ij', 'lat_lon']:
            llc = (self.y.min(), self.x.min())
        else:
            raise ValueError(
                "Unrecognised 'order' parameter: {}".format(order)
            )
        return llc

    def get_urc(self, order='xy'):
        if order in ['xy', 'ji', 'lon_lat']:
            urc = (self.x.max(), self.y.max())
        elif order in ['yx', 'ij', 'lat_lon']:
            urc = (self.y.max(), self.x.max())
        else:
            raise ValueError(
                "Unrecognised 'order' parameter: {}".format(order)
            )
        return urc

    def index_to_coords(self, i, j, order='ij'):
        if order == 'xy':
            i, j = j, i
        elif order != 'ij':
            raise ValueError(
                "Unrecognised 'order' parameter: {}".format(order)
            )
        x = self.x[i, j]
        y = self.y[i, j]
        return x, y

    @property
    def x(self):
        '''
        Docstring for x goes here.
        '''
        return self._x

    @property
    def y(self):
        '''
        Docstring for y goes here.
        '''
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def nx(self):
        '''
        Docstring for nx goes here.
        '''
        return self._nx

    @property
    def ny(self):
        '''
        Docstring for ny goes here.
        '''
        return self._ny

    @property
    def input_topo(self):
        '''
        Docstring for input_topo goes here.
        '''
        return self._input_topo

    @property
    def tin(self):
        '''
        Docstring for tin goes here.
        '''
        return self._tin

    @tin.setter
    def tin(self, tin_file):
        if tin_file is None:
            self._tin = None
            self._sealevel = 0.0
        else:
            xmf_file = os.path.join(
                os.path.dirname(tin_file),
                '..', 'xmf',
                os.path.basename(tin_file).replace('.hdf5', '.xmf'),
            )
            try:
                self._sealevel = get_sealevel(xmf_file)
            except (
                FileNotFoundError,
                TypeError,
                AttributeError,
                ValueError,
            ):
                self._sealevel = 0.0
            rXY = np.hstack((
                self.x.reshape((-1, 1)),
                self.y.reshape((-1, 1)),
            ))
            self._tin = _GridTIN(tin_file, rXY, self.nx, self.ny)

    @property
    def sealevel(self):
        '''
        Sealevel at the given timestep.
        '''
        return self._sealevel

    @property
    def flow(self):
        '''
        Docstring for flow goes here.
        '''
        return self._flow

    @flow.setter
    def flow(self, flow_file):
        if flow_file is None:
            self._flow = None
        else:
            rXY = np.hstack((
                self.x.reshape((-1, 1)),
                self.y.reshape((-1, 1)),
            ))
            self._flow = _GridFlow(flow_file, rXY, self.nx, self.ny)

    @property
    def strat(self):
        '''
        Docstring for strat goes here.
        '''
        return self._strat

    @strat.setter
    def strat(self, sed_file):
        if sed_file is None:
            self._strat = None
        else:
            rXY = np.hstack((
                self.x.reshape((-1, 1)),
                self.y.reshape((-1, 1)),
            ))
            self._strat = _GridStrat(sed_file, rXY, self.nx, self.ny)


class _GridData:
    '''
    Base structure for holding data within a Grid instance.
    '''
    def __init__(self, dtype, filename=None):
        if dtype not in ['tin', 'flow', 'strat']:
            raise ValueError("Unrecognised data type: '{}'".format(dtype))
        self.dtype = dtype
        self.filename = filename

    def _plot(self, data, ax=None, **kwargs):
        '''
        Generic plotting functionality.

        --------------------------------------------------------------------------
        Parameters (optional):
        --------------------------------------------------------------------------
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        if ax is None:
            plt.imshow(data, **kwargs)
        else:
            ax.imshow(data, **kwargs)


class _GridTIN(_GridData):
    '''
    Structure for holding TIN-related data within
    a Grid instance.
    '''
    def __init__(self, tin_file, rXY, nx, ny):
        super().__init__('tin', tin_file)
        try:
            if not tin_file.endswith('.hdf5'):
                tin_file += '.hdf5'
        except AttributeError as e:
            raise TypeError("Invalid filename: {}".format(tin_file)) from e
        if not os.path.isfile(tin_file):
            raise FileNotFoundError("File not found: {}".format(tin_file))
        data = grid_tools.mapTIN2Reg(tin_file, rXY, nx, ny)

        self.z = data['z']
        self.area = data['area']
        self.discharge = data['discharge']
        self.cumdiff = data['cumdiff']
        self.cumfail = data['cumfail']
        self.cumflex = data['cumflex']
        self.cumhill = data['cumhill']

    def plot_elevation(self, ax=None, **kwargs):
        '''
        Plot elevation.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.z, ax, **kwargs)

    def plot_erodep(self, ax=None, **kwargs):
        '''
        Plot cumulative erosion/deposition.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.cumdiff, ax, **kwargs)

    def plot_discharge(self, ax=None, **kwargs):
        '''
        Plot discharge.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.discharge, ax, **kwargs)

    def plot_failure(self, ax=None, **kwargs):
        '''
        Plot cumulative hillslope failure.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.cumfail, ax, **kwargs)

    def plot_flexure(self, ax=None, **kwargs):
        '''
        Plot cumulative flexural isostasy.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.cumflex, ax, **kwargs)

    def plot_hillcreep(self, ax=None, **kwargs):
        '''
        Plot cumulative hillslope diffusion (creep).
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.cumhill, ax, **kwargs)


class _GridFlow(_GridData):
    '''
    Structure for holding flow-related data within
    a Grid instance.
    '''
    def __init__(self, flow_file, rXY, nx, ny):
        super().__init__('flow', flow_file)
        try:
            if not flow_file.endswith('.hdf5'):
                flow_file += '.hdf5'
        except AttributeError as e:
            raise TypeError("Invalid filename: {}".format(flow_file)) from e
        if not os.path.isfile(flow_file):
            raise FileNotFoundError("File not found: {}".format(flow_file))
        data = grid_tools.mapTIN2Reg(flow_file, rXY, nx, ny)

        self.basin = data['basin']
        self.chi = data['chi']
        self.flowdensity = data['flowdensity']
        self.sedload = data['sedload']

    def plot_basin(self, ax=None, **kwargs):
        '''
        Plot 'basin' parameter.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.basin, ax, **kwargs)

    def plot_chi(self, ax=None, **kwargs):
        '''
        Plot 'chi' parameter.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.chi, ax, **kwargs)

    def plot_flowdensity(self, ax=None, **kwargs):
        '''
        Plot flow density.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.flowdensity, ax, **kwargs)

    def plot_sedload(self, ax=None, **kwargs):
        '''
        Plot sediment load.
        ----------------------------------------------------------------------
        Parameters (optional):
            ax: a matplotlib.axes.Axes instance on which to draw
                the image.
            **kwargs: further kwargs are passed to ax.imshow.
        '''
        self._plot(self.sedload, ax, **kwargs)


class _GridStrat(_GridData):
    '''
    Structure for holding stratigraphy-related data within
    a Grid instance
    '''
    def __init__(self, sed_file, rXY, nx, ny):
        super().__init__('strat', sed_file)
        try:
            if not sed_file.endswith('.hdf5'):
                sed_file += '.hdf5'
        except AttributeError as e:
            raise TypeError("Invalid filename: {}".format(sed_file)) from e
        if not os.path.isfile(sed_file):
            raise FileNotFoundError("File not found: {}".format(sed_file))
        with h5py.File(sed_file, mode='r') as f:
            coords = np.asarray(f['coords'])
            layDepth = np.asarray(f['layDepth'])
            layElev = np.asarray(f['layElev'])
            layThick = np.asarray(f['layThick'])
        tree = cKDTree(coords)
        inds = tree.query(rXY, k=1)[1]
        rdepth = layDepth[inds].reshape((ny, nx, -1))
        relev = layElev[inds].reshape((ny, nx, -1))
        rthick = layThick[inds].reshape((ny, nx, -1))

        self.lay_depth = rdepth
        self.lay_elev = relev
        self.lay_thick = rthick


if __name__ == '__main__':
    # Testing
    test_llc = (-6, 18)
    test_urc = (35, 50)
    test_UTMzone = 36
    test_projection = Proj(
        "+proj=utm +zone={}, +north ".format(test_UTMzone)
        + "+ellps=WGS84 +datum=WGS84 "
        + "+units=m +no_defs"
    )
    test_resolution = 7500.
    test_gb = GridBuilder(test_llc, test_urc, test_resolution, test_projection)
