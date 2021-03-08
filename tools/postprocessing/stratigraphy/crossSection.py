import time

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .sectionBase import _SectionBase
from .wheeler import WheelerDiagram
from ...grid import Grid
from ...io import hdf5
from ...tools.misc import array_average, format_time

# TODO: Add documentation


class CrossSection(_SectionBase):
    '''
    Class for creating cross-sections from Badlands outputs.
    '''

    def __init__(
        self,
        a,
        b,
        input_topo=None,
        sed_file=None,
        order='ij',
        num_points=None,
        units='km',
        **kwargs
    ):
        '''
        Create a CrossSection instance from endpoints and a base topography
        file.

        ----------------------------------------------------------------------
        Parameters:
        ----------------------------------------------------------------------
            a: the (i, j) or (x, y) coordinate tuple (in grid indices) of the
                start point of the cross-section.
            b: the (i, j) or (x, y) coordinate tuple (in grid indices) of the
                end point of the cross-section.
            input_topo (optional): path to the .csv file defining the regular
                Badlands grid. If not given now, it must be set later.
                (default None)
            sed_file (optional): the stratigraphy Badlands output file
                to be used to produce the cross-section. If not given now,
                it must be set later.
                (default None)
            order (optional): the order ('ij' or 'xy') of the coordinates
                given in a and b.
                (default 'ij')
            num_points
            units
        '''
        super().__init__(a, b, input_topo, order)

        self.sed_file = sed_file

        self._units = units
        if num_points is not None:
            self._dists = self.get_dists(num_points, units)
        else:
            self._dists = None
        self._depths = None
        self._elevations = None
        self._thicknesses = None

        if num_points is not None:
            self.generate_section(
                num_points,
                verbose=kwargs.get('verbose', False),
            )

    @classmethod
    def from_xy(cls, a, b, input_topo, sed_file=None, order='xy'):
        '''
        Create a CrossSection instance with endpoints defined in projected
        coordinates, rather than indices on the regular grid.

        ----------------------------------------------------------------------
        Parameters:
        ----------------------------------------------------------------------
            a: the (x, y) or (y, x) projected coordinate tuple (in metres)
                representing the start point of the cross-section.
            b: the (x, y) or (y, x) projected coordinate tuple (in metres)
                representing the end point of the cross-section.
            input_topo: path to the .csv file defining the regular Badlands
                grid.
            sed_file (optional): the stratigraphy Badlands output file
                to be used to produce the cross-section. If not given now,
                it must be set later.
                (default None)
            order (optional): the order ('xy' or 'yx') of the coordinates
                given in a and b.
                (default 'xy')

        Returns:
        ----------------------------------------------------------------------
            cs: the created CrossSection instance.
        '''
        if order == 'yx':
            a = a[::-1]
            b = b[::-1]
        elif order != 'xy':
            raise ValueError("Unrecognised 'order' parameter: {}.".format(order))
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
        cs = cls(a, b, input_topo, sed_file, order='xy')
        return cs

    def generate_section(self, num_points, **kwargs):
        '''
        Generate the cross-section using scipy.ndimage.map_coordinates.

        ----------------------------------------------------------------------
        Parameters:
            num_points: number of points to use for the generated
                cross-section.
            **kwargs: valid kwargs include:
                unit: unit ('m' or 'km') to use for the generated values.
                    (default 'km')
                verbose: if True, print more information.
                    (default False)
        '''
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('Building cross-section '
                  + '(num. points = {})...'.format(num_points))
        tstart = time.time()
        points = self._section_points(num_points)
        dists = self._section_dists(num_points)
        _, rdepths, relevations, rthicknesses = self._map_sed_to_grid()
        cs_depths = np.zeros((rdepths.shape[-1], num_points))
        cs_elevations = np.zeros((relevations.shape[-1], num_points))
        cs_thicknesses = np.zeros((rthicknesses.shape[-1], num_points))
        for i in range(rdepths.shape[-1]):
            if verbose:
                print(
                    '\tLayer {} of {}...'.format(i + 1, rdepths.shape[-1]),
                    end='',
                )
            cs_depths[i] = map_coordinates(rdepths[..., i], points)
            cs_elevations[i] = map_coordinates(relevations[..., i], points)
            cs_thicknesses[i] = map_coordinates(rthicknesses[..., i], points)
            if verbose:
                print('\r', end='')
        if verbose:
            print('')
        for i in range(1, cs_depths.shape[0]):
            mask = cs_depths[i] < cs_depths[i - 1]
            cs_depths[i][mask] = cs_depths[i - 1][mask]
        if kwargs.get('unit', 'km') == 'km':
            dists *= 1.e-3
            cs_depths *= 1.e-3
            cs_elevations *= 1.e-3
            cs_thicknesses *= 1.e-3

        cs_depths = cs_depths[np.all(cs_thicknesses != 0, axis=1)]
        cs_elevations = cs_elevations[np.all(cs_thicknesses != 0, axis=1)]
        cs_thicknesses = cs_thicknesses[np.all(cs_thicknesses != 0, axis=1)]

        self._units = kwargs.get('unit', 'km')
        self._dists = dists
        self._depths = cs_depths
        self._elevations = cs_elevations
        self._thicknesses = cs_thicknesses

        if verbose:
            tend = time.time()
            duration = tend - tstart
            duration = format_time(duration)
            print("Cross-section built; duration = {}".format(duration))

    def _check_built(self):
        for i in (
            self.dists,
            self.depths,
            self.elevations,
            self.thicknesses,
        ):
            if i is None:
                raise AttributeError('Cross-section has not yet been built')

    def get_sed_flux(self, dt=1.0):
        self._check_built()
        dt = float(dt)
        sed_flux = np.asarray(
            self.thicknesses.sum(axis=1)
            * (self.dists[1] - self.dists[0])
            / dt
        )
        return sed_flux

    def _get_shelf_edges(
        self,
        critical_slope=-0.025,
        degrees=True,
        interval=1
    ):
        self._check_built()
        if degrees:
            critical_slope = np.tan(np.radians(critical_slope))
        interval = int(interval)
        if interval < 0:
            raise ValueError('Interval less than one: {}'.format(interval))
        if interval > 1:
            grad = np.gradient(
                array_average(self.elevations, N=interval),
                axis=1,
            )
        else:
            grad = np.gradient(self.elevations, axis=1)
        grad /= np.abs(self.dists[1] - self.dists[0])
        data = {}
        rows, columns = np.where(grad < critical_slope)
        for i, j in zip(rows, columns):
            if (
                (i in data and j < data[i])
                or (i not in data)
            ):
                data[i] = j
        return data

    def get_shelf_edges(
        self,
        critical_slope=-0.025,
        degrees=True,
        interval=1,
    ):
        '''
        Find the location of the shelf edge through time.

        ----------------------------------------------------------------------
        Parameters:
        ----------------------------------------------------------------------
            critical_slope
            degrees
            interval

        Returns:
        ----------------------------------------------------------------------
            locs: (N, 2) array of (x, y) coordinate pairs.
        '''
        self._check_built()
        data = self._get_shelf_edges(critical_slope, degrees, interval)
        y0 = sorted(data.keys())
        x0 = [data[key] for key in y0]
        x = np.asarray(self.dists[x0])
        y = np.asarray(array_average(self.depths, interval)[y0, x0])
        locs = np.hstack((
            x.reshape((-1, 1)),
            y.reshape((-1, 1)),
        ))
        return locs

    def plot_shelf_edges(
        self,
        ax=None,
        critical_slope=-0.025,
        interval=1,
        **kwargs
    ):
        if ax is None:
            func = plt.plot
        else:
            func = ax.plot
        if 'degrees' in kwargs:
            degrees = kwargs.pop('degrees')
        else:
            degrees = False
        locs = self.get_shelf_edges(critical_slope, degrees, interval)
        x = locs[:, 0]
        y = locs[:, 1]
        p = func(x, y, **kwargs)
        return p

    def scatter_shelf_edges(
        self,
        ax=None,
        critical_slope=-0.025,
        interval=1,
        plot_interval=1,
        **kwargs
    ):
        if ax is None:
            func = plt.scatter
        else:
            func = ax.scatter
        if 'degrees' in kwargs:
            degrees = kwargs.pop('degrees')
        else:
            degrees = False
        plot_interval = int(plot_interval)
        locs = self.get_shelf_edges(critical_slope, degrees, interval)
        x = locs[:, 0]
        y = locs[:, 1]
        p = func(x[::plot_interval], y[::plot_interval], **kwargs)
        return p


    def plot_section(self, ax=None, fill=False, **kwargs):
        self._check_built()
        kwargs = kwargs.copy()
        if ax is None:
            func = plt.plot
            func2 = plt.fill_between
        else:
            func = ax.plot
            func2 = ax.fill_between

        # Fill parameters
        if 'colours' in kwargs:
            colours = kwargs.pop('colours')
        else:
            colours = None
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
            cmap = cm.get_cmap(cmap)
        elif fill:
            cmap = 'RdYlBu'
            cmap = cm.get_cmap(cmap)
        if 'vmin' in kwargs:
            vmin = float(kwargs.pop('vmin'))
        else:
            vmin = 0.0
        if 'vmax' in kwargs:
            vmax = float(kwargs.pop('vmax'))
        else:
            vmax = float(self.depths.shape[0])
        if 'fill_alpha' in kwargs:
            fill_alpha = float(kwargs.pop('fill_alpha'))
        else:
            fill_alpha = 1.0

        # Other plotting parameters
        if 'interval' in kwargs:
            interval = int(kwargs.pop('interval'))
        else:
            interval = 1
        if 'xmin' in kwargs:
            xmin = float(kwargs.pop('xmin'))
        else:
            xmin = self.dists.min()
        if 'xmax' in kwargs:
            xmax = float(kwargs.pop('xmax'))
        else:
            xmax = self.dists.max()
        if ('ymin' in kwargs) and ('ymax' in kwargs):
            ymin = float(kwargs.pop('ymin'))
            ymax = float(kwargs.pop('ymax'))
        else:
            tmp = self.depths[
                :,
                np.logical_and(
                    self.dists >= xmin,
                    self.dists <= xmax,
                )
            ]
            yrange = tmp.max() - tmp.min()
            if 'ymin' in kwargs:
                ymin = float(kwargs.pop('ymin'))
            else:
                ymin = tmp.min() - (yrange * 0.05)
            if 'ymax' in kwargs:
                ymax = float(kwargs.pop('ymax'))
            else:
                ymax = tmp.max() + (yrange * 0.05)

        # Plot appearance parameters
        if 'background_colour' in kwargs:
            background_colour = str(kwargs.pop('background_colour'))
        else:
            background_colour = None
        if 'sea_colour' in kwargs:
            sea_colour = str(kwargs.pop('sea_colour'))
        else:
            sea_colour = None
        if 'sea_level' in kwargs:
            sea_level = float(kwargs.pop('sea_level'))
        else:
            sea_level = 0.0

        j = self.depths.shape[0]
        k = 0
        for i in range(0, self.depths.shape[0], interval):
            func(self.dists, self.depths[i, ...], zorder=100, **kwargs)
            if (fill) and (i != 0):
                if colours is not None:
                    if isinstance(colours, str):
                        colour = colours
                    else:
                        colour = colours[k]
                else:
                    colour = cmap((j - vmin) / vmax)
                func2(
                    self.dists,
                    self.depths[i - interval, ...],
                    self.depths[i, ...],
                    color=colour,
                    alpha=fill_alpha,
                    zorder=5,
                )
            j -= interval
            k += 1

        if background_colour is not None:
            func2(
                self.dists,
                -1.e10, self.depths[0, ...],
                color=background_colour, zorder=1,
                alpha=0.5,
            )
        if sea_colour is not None:
            tmpx = self.dists[self.depths[-1, ...] < sea_level]
            tmpy0 = self.depths[-1, ...][self.depths[-1, ...] < sea_level]
            tmpy1 = np.asarray([sea_level] * len(tmpx))
            func2(tmpx, tmpy0, tmpy1, alpha=0.25, zorder=1, color=sea_colour)
            func(tmpx, tmpy1, color=sea_colour, zorder=1)

        if ax is None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if fill:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin, vmax)
            )
            return sm

    def plot_section_depenv(
        self,
        ax=None,
        colours=None,
        plot_kwargs=None,
        fill_kwargs=None,
        **kwargs
    ):
        '''
        Plot the cross-section, with layers coloured according to the
        depth of deposition (a proxy for depositional environment).

        ----------------------------------------------------------------------
        Parameters:
        ----------------------------------------------------------------------
            ax:
            colours:
            plot_kwargs:
            fill_kwargs:
            **kwargs:
        '''
        self._check_built()
        if ax is None:
            func = plt.plot
            func2 = plt.fill_between
        else:
            func = ax.plot
            func2 = ax.fill_between

        # Process kwargs
        if plot_kwargs is None:
            plot_kwargs = {}
        if fill_kwargs is None:
            fill_kwargs = {}
        interval = int(kwargs.get('interval', 1))
        xmin = float(kwargs.get('xmin', self.dists.min()))
        xmax = float(kwargs.get('xmax', self.dists.max()))
        if ('ymin' in kwargs) and ('ymax' in kwargs):
            ymin = float(kwargs['ymin'])
            ymax = float(kwargs['ymax'])
        else:
            tmp = self.depths[
                :,
                np.logical_and(
                    self.dists >= xmin,
                    self.dists <= xmax,
                )
            ]
            yrange = tmp.max() - tmp.min()
            ymin = float(kwargs.get('ymin', tmp.min() - (yrange * 0.05)))
            ymax = float(kwargs.get('ymax', tmp.max() + (yrange * 0.05)))
        background_colour = str(kwargs.get('background_colour', None))
        sea_colour = str(kwargs.get('sea_colour', None))
        sea_level = float(kwargs.get('sea_level', 0.0))

        # Draw layers
        for i in range(0, self.depths.shape[0], interval):
            func(self.dists, self.depths[i, ...], zorder=100, **plot_kwargs)

        # Fill layers
        wheel = self.wheeler(
            colours=colours,
            minimum_thickness=0.0,
            interval=interval,
        )
        d = wheel.get_colours()
        cmap = d['cmap']
        vmin = d['vmin']
        vmax = d['vmax']
        if 'zorder' in fill_kwargs:
            zorder = float(fill_kwargs.pop('zorder'))
        else:
            zorder = 50.0
        j = 0
        for i in range(interval, self.depths.shape[0], interval):
            # for val in np.unique(wheel.array[i, ...]):
            for val in np.unique(wheel.array[j, ...]):
                func2(
                    self.dists,
                    self.depths[i - interval, ...],
                    self.depths[i, ...],
                    # where=(wheel.array[i, ...] >= val),
                    where=(wheel.array[j, ...] >= val),
                    color=cmap((val - vmin) / (vmax - vmin)),
                    zorder=zorder + val,
                    **fill_kwargs
                )
            j += 1

        if background_colour is not None:
            func2(
                self.dists,
                -1.e10, self.depths[0, ...],
                color=background_colour, zorder=1,
                alpha=0.5,
            )
        if sea_colour is not None:
            tmpx = self.dists[self.depths[-1, ...] < sea_level]
            tmpy0 = self.depths[-1, ...][self.depths[-1, ...] < sea_level]
            tmpy1 = np.asarray([sea_level] * len(tmpx))
            func2(tmpx, tmpy0, tmpy1, alpha=0.25, zorder=1, color=sea_colour)
            func(tmpx, tmpy1, color=sea_colour, zorder=1)

        if ax is None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    def wheeler(
        self,
        colours=None,
        minimum_thickness=None,
        interval=1,
    ):
        '''
        Create a Wheeler diagram for the given section.

        ----------------------------------------------------------------------
        Parameters:
        ----------------------------------------------------------------------
            colours: a template dictionary for the Wheeler diagram (see
                paleoflow.postprocessing.stratigraphy.DEFAULT_WHEELER for
                an example).
                (default paleoflow.postprocessing.stratigraphy.DEFAULT_WHEELER)
            minimum_thickness: minimum thickness (in metres) for layers to
                be drawn on the Wheeler diagram.
                (default 0.5)
            interval:
                (default 1)

        Returns:
        ----------------------------------------------------------------------
            wheel: a paleoflow.postprocessing.stratigraphy.Wheeler instance.
        '''
        self._check_built()
        if minimum_thickness is None:
            minimum_thickness = 0.5
            # if self.units == 'km':
                # minimum_thickness *= 1.e-3
        arr1 = self.elevations.copy()
        arr2 = self.thicknesses.copy()
        if interval != 1:
            arr1 = array_average(arr1, N=interval)
            arr2 = array_average(arr2, N=interval)

        if self.units == 'km':
            arr1 *= 1.e3
            arr2 *= 1.e3
        arr2[arr2 < 0] = 0.0
        wheel = WheelerDiagram(
            arr1,
            arr2,
            colours,
            minimum_thickness,
        )
        return wheel


    def _map_sed_to_grid(self):
        coords, layDepth, layElev, layThick = self._read_sed()
        grid = Grid(self.input_topo)
        rXY = grid.get_rXY()
        tree = cKDTree(coords)
        _, inds = tree.query(rXY, k=1)
        depths = layDepth[inds].reshape((grid.ny, grid.nx, -1))
        elevations = layElev[inds].reshape((grid.ny, grid.nx, -1))
        thicknesses = layThick[inds].reshape((grid.ny, grid.nx, -1))
        return coords, depths, elevations, thicknesses

    def _read_sed(self):
        coords, layDepth, layElev, layThick = hdf5.read_sed(self.sed_file)
        return coords, layDepth, layElev, layThick

    @property
    def units(self):
        '''
        The units (m or km) of distances used for the section.
        '''
        return self._units

    @property
    def dists(self):
        '''
        The array of distances along the cross-section.
        '''
        return self._dists

    @property
    def depths(self):
        '''
        The array of stratum depths along the cross-section.
        '''
        return self._depths

    @property
    def elevations(self):
        '''
        The array of stratum paleo-elevations along the cross-section.
        '''
        return self._elevations

    @property
    def thicknesses(self):
        '''
        The array of stratum thicknesses along the cross-section.
        '''
        return self._thicknesses
