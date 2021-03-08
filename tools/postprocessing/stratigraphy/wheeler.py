from collections import OrderedDict
import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from ...io import nc

DEFAULT_WHEELER = {
    'None': 'white',
    (np.inf, 0.0): 'limegreen',
    (0.0, -30.0): 'darkkhaki',
    (-30.0, -100.0): 'sandybrown',
    (-100.0, -300.0): 'khaki',
    (-300.0, -500.0): 'cyan',
    (-500.0, -np.inf): 'teal',
}
DEFAULT_WHEELER_UNITS = 'm'


def _default_wheeler():
    return DEFAULT_WHEELER.copy()


def _process_wheeler_dict(colours=None):
    if colours is None:
        colours = _default_wheeler()
    colours = colours.copy()
    none_colour = colours['None']
    colours.pop('None')
    colours = OrderedDict(
        sorted(colours.items(), key=lambda t: -1 * t[0][0])
    )
    new_values = list(range(len(colours)))
    d_colours = {-1: none_colour}
    d_values = {'None': -1}
    for i, j in enumerate(colours):
        if j == 'key':
            continue
        d_colours[new_values[i]] = colours[j]
        d_values[j] = new_values[i]
    return d_colours, d_values


def _get_wheeler_array(arr, colours=None):
    if colours is None:
        colours = _default_wheeler()
    _, d_values = _process_wheeler_dict(colours)
    arr_out = np.full(arr.shape, -1, dtype=int)
    i = 0
    for key in d_values:
        if key == 'None':
            continue
        x0, x1 = key
        mask = np.logical_and(arr < x0, arr >= x1)
        arr_out[mask] = i
        i += 1
    return arr_out


def concatenate(*args):
    arrays = []
    for arg in args:
        if not isinstance(arg, WheelerDiagram):
            raise TypeError(
                "All arguments must be a WheelerDiagram instance"
            )
        try:
            if colours != arg.colours:
                raise ValueError(
                    "All arguments must have the same 'colours' attribute"
                )
        except NameError:
            pass
        colours = arg.colours
        array = arg.array
        arrays.append(array)
    array = np.vstack(arrays)
    WheelerDiagram.from_array(array, colours=colours)


class WheelerDiagram:
    def __init__(
        self,
        elevations=None,
        thicknesses=None,
        colours=None,
        minimum_thickness=None,
        array=None,
    ):
        if colours is None:
            colours = _default_wheeler()
        if minimum_thickness is None:
            minimum_thickness = 0.5
        if (elevations is not None) and (thicknesses is not None):
            if array is not None:
                raise ValueError(
                    "Cannot provide 'elevations', 'thicknesses', and 'array'"
                )
            wheeler_arr = _get_wheeler_array(elevations, colours=colours)
            wheeler_arr[thicknesses < minimum_thickness] = -1
        elif array is not None:
            wheeler_arr = array.copy()
        else:
            raise ValueError(
                "Must provide either 'elevations' and 'thicknesses' "
                + "or 'array'"
            )
        self._array = wheeler_arr
        self._colours = colours
        self._cmap = self._create_cmap()

    @classmethod
    def from_array(cls, array, colours=None):
        wheel = cls(
            colours=colours,
            array=array,
        )
        return wheel

    def show(self, ax=None, **kwargs):
        kw = dict(
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            **self.get_colours(),
        )
        kwargs = kwargs.copy()
        for key in kw:
            if key in kwargs:
                kwargs.pop(key)
        if ax is None:
            func = plt.imshow
        else:
            func = ax.imshow
        im = func(self.array, **kwargs, **kw)
        return im

    def get_colours(self):
        '''
        Get cmap, vmin, and vmax for plotting the Wheeler diagram.

        ----------------------------------------------------------------------
        Returns:
        ----------------------------------------------------------------------
            d: dict with the following keys: 'cmap', 'vmin', and 'vmax'.
        '''
        d_colours, _ = _process_wheeler_dict(self.colours)
        d = {
            'cmap': self.cmap,
            'vmin': -1.5,
            'vmax': len(d_colours) - 1.5,
        }
        return d

    def to_file(self, filename, verbose=False):
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            raise FileNotFoundError(
                "Output directory does not exist: '{}'".format(dirname)
            )
        x = np.arange(self.array.shape[1])
        y = np.arange(self.array.shape[0])
        nc.write(self.array, x, y, filename, verbose)

    def _create_cmap(self):
        d_colours, _ = _process_wheeler_dict(self.colours)
        colours_list = []
        for _, value in sorted(d_colours.items(), key=lambda t: t[0]):
            colours_list.append(value)
        cmap = cm.colors.ListedColormap(colours_list)
        return cmap

    @property
    def array(self):
        return self._array

    @property
    def colours(self):
        return self._colours

    @property
    def cmap(self):
        return self._cmap
