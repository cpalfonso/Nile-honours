import os

import h5py
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ..io import ENCODING
from ..tools import projection as projection_tools


def get_grid_extents(
        llc,
        urc,
        projection,
        order='lat_lon',
        verbose=False,
):
    '''
    Calculate the extents (UTM easting, northing) of the Badlands grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        llc: (lat, lon) or (lon, lat) of the lower left corner.
        urc: (lat, lon) or (lon, lat) of the upper right corner.
        resolution: resolution of the grid (in metres).
        projection: a pyproj.Proj object or a pyproj string.
        order: 'lat_lon' or 'lon_lat' (default 'lat_lon').

    Returns:
    --------------------------------------------------------------------------
        easting, northing: (x_min, x_max), (y_min, y_max) of the Badlands
            grid (in projected UTM coordinates).
    '''

    # Process 'order' parameter
    if order == 'lat_lon':
        for i in [llc, urc]:
            projection_tools.test_valid_lat_lon(i)
    elif order == 'lon_lat':
        for i in [llc, urc]:
            projection_tools.test_valid_lon_lat(i)
        llc = (llc[1], llc[0])
        urc = (urc[1], urc[0])
    else:
        raise ValueError("Invalid 'order' parameter: {}".format(order))

    # Projection:
    if isinstance(projection, str):
        projection = Proj(projection)

    # Grid extents:
    UTMx1, UTMy1 = projection(llc[1], llc[0])
    UTMx2, UTMy2 = projection(urc[1], urc[0])
    if verbose:
        print(
            'UTM eastings: {} / {}'.format(round(UTMx1, 0), round(UTMx2, 0))
        )
        print(
            'UTM northings: {} / {}'.format(round(UTMy1, 0), round(UTMy2, 0))
        )
    easting = [round(UTMx1, -3), round(UTMx2, -3)]
    northing = [round(UTMy1, -3), round(UTMy2, -3)]

    return easting, northing


def get_grid_dimensions(
        easting=None,
        northing=None,
        llc=None,
        urc=None,
        order=None,
        resolution=None,
        projection=None,
        verbose=False,
):
    '''
    Calculates the dimensions of the grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        resolution: resolution of the grid (in metres)
        AND ONE OF:
        easting, northing: (x_min, x_max), (y_min, y_max) of the Badlands
            grid (in projected UTM coordinates)
        OR
        llc, urc: (lat, lon) or (lon, lat) of the lower left and upper
            right corners
        projection: a pyproj.Proj object or a pyproj string

    Returns:
    --------------------------------------------------------------------------
        nx, ny: the x and y dimensions of the grid
    '''
    # Check for valid input combination:
    error_message = (
        "Invalid input combination (choose easting, northing "
        + "or llc, urc)"
    )
    option_1 = easting and northing
    option_2 = llc and urc
    if easting or northing:
        if not option_1:
            raise ValueError(error_message)
    if llc or urc:
        if not option_2:
            raise ValueError(error_message)

    # Process 'order' parameter
    if option_2:
        if order is None:
            order = 'lat_lon'
        if order == 'lat_lon':
            for i in [llc, urc]:
                projection_tools.test_valid_lat_lon(i)
        elif order == 'lon_lat':
            for i in [llc, urc]:
                projection_tools.test_valid_lon_lat(i)
            llc = (llc[1], llc[0])
            urc = (urc[1], urc[0])
        else:
            raise ValueError("Invalid 'order' parameter: {}".format(order))

        easting, northing = get_grid_extents(
            llc,
            urc,
            projection,
            verbose=verbose,
        )

    resbad = float(resolution)
    xcoords = np.arange(easting[0], easting[1], resbad)
    nx = len(xcoords)
    ycoords = np.arange(northing[0], northing[1], resbad)
    ny = len(ycoords)

    return nx, ny


def get_grid_nodes(
        easting=None,
        northing=None,
        llc=None,
        urc=None,
        order=None,
        resolution=None,
        projection=None,
        return_XYcoords=False,
        verbose=True,
):
    '''
    Returns the (projected UTM) coordinates of the grid nodes.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        resolution: resolution of the grid (in metres)
        projection: a pyproj.Proj object or a pyproj string
        AND ONE OF:
        easting, northing: (x_min, x_max), (y_min, y_max) of the Badlands
            grid (in projected UTM coordinates)
        OR
        llc, urc: (lat, lon) or (lon, lat) of the lower left
            and upper right corners

    Returns:
    --------------------------------------------------------------------------
        mlon:
        mlat:
        nx:
        ny:
        XYcoords (optional):
    '''
    # Check for valid input combination:
    error_message = (
        "Invalid input combination (choose easting, northing "
        + "or llc, urc)"
    )
    option_1 = easting and northing
    option_2 = llc and urc
    if easting or northing:
        if not option_1:
            raise ValueError(error_message)
    if llc or urc:
        if not option_2:
            raise ValueError(error_message)

    # Process 'order' parameter
    if option_2:
        if order is None:
            order = 'lat_lon'
        if order == 'lat_lon':
            for i in [llc, urc]:
                projection_tools.test_valid_lat_lon(i)
        elif order == 'lon_lat':
            for i in [llc, urc]:
                projection_tools.test_valid_lon_lat(i)
            llc = (llc[1], llc[0])
            urc = (urc[1], urc[0])
        else:
            raise ValueError("Invalid 'order' parameter: {}".format(order))

        easting, northing = get_grid_extents(llc, urc, projection, order=order)

    resbad = float(resolution)
    xcoords = np.arange(easting[0], easting[1], resbad)
    ycoords = np.arange(northing[0], northing[1], resbad)
    X, Y = np.meshgrid(xcoords, ycoords)
    nx = X.shape[1]
    ny = X.shape[0]
    if verbose:
        # Information about the grid:
        print('Badlands mesh size: {} x {}'.format(nx, ny))
        print('Badlands grid resolution: {} m'.format(resbad))
    XYcoords = np.vstack([X.ravel(), Y.ravel()]).T
    mlat = np.zeros(len(XYcoords))
    mlon = np.zeros(len(XYcoords))
    mlon, mlat = projection(XYcoords[:, 0], XYcoords[:, 1], inverse=True)

    if return_XYcoords:
        return (mlon, mlat, nx, ny, XYcoords)

    return (mlon, mlat, nx, ny)


def build_grid(
        llc,
        urc,
        resolution,
        projection,
        order='lat_lon',
        return_XYcoords=False,
        verbose=True
):
    '''
    High-level function for building a Badlands input grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        llc: (lat, lon) or (lon, lat) of the lower left corner.
        urc: (lat, lon) or (lon, lat) of the upper right corner.
        resolution: resolution of the grid (in metres).
        projection: a pyproj.Proj object or a pyproj string.
        order: 'lat_lon' or 'lon_lat'.

    Returns:
    --------------------------------------------------------------------------
        mlon, mlat: the longitude and latitude points of the Badlands
            grid.
        nx, ny: the number of x and y points in the Badlands grid.
        XYcoords (optional): the x, y points of the Badlands grid.
    '''

    # Process 'order' parameter
    if order == 'lat_lon':
        for i in [llc, urc]:
            projection_tools.test_valid_lat_lon(i)
    elif order == 'lon_lat':
        for i in [llc, urc]:
            projection_tools.test_valid_lon_lat(i)
        llc = (llc[1], llc[0])
        urc = (urc[1], urc[0])
    else:
        raise ValueError("Invalid 'order' parameter: {}".format(order))

    # Projection:
    if type(projection) == str:
        projection = Proj(projection)

    # Grid extents:
    easting, northing = get_grid_extents(
        llc,
        urc,
        projection,
        verbose=verbose,
    )

    # Build mesh:
    resbad = float(resolution)
    mlon, mlat, nx, ny, XYcoords = get_grid_nodes(
        easting=easting,
        northing=northing,
        resolution=resbad,
        projection=projection,
        return_XYcoords=True,
        verbose=verbose,
    )

    # Check the size matches:
    width = np.ceil((easting[1] - easting[0]) / resbad)
    height = np.ceil((northing[1] - northing[0]) / resbad)
    if width != nx or height != ny:
        print('Warning: inconsistency detected:')
        if width != nx:
            print('width = {}, nx = {}'.format(width, nx))
        if height != ny:
            print('height = {}, ny = {}'.format(height, ny))

    if return_XYcoords:
        return (mlon, mlat, nx, ny, XYcoords)

    return (mlon, mlat, nx, ny)


def remap_TIN(
        input_topo,
        step,
        badlands_out,
        return_all=False,
        savegrid=None,
        savefig=None,
        verbose=True,
):
    '''
    Remaps irregular TIN data to a regular grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        input_topo: the .csv file containing the initial topography.
            This will be used to map the irregular TIN output to a
            regular grid.
        step: the frame number of the final Badlands output step
            (usually equal to (end_time-start_time)/step_time).
        badlands_out: the output directory of the Badlands model.
        return_all: if True, will return all arrays in the hdf5 file,
            rather than just the elevation (default False).
        savegrid: the filename to save the paleotopography in .csv
            format (default None).
        savefig: the filename to save an image of the paleotopography
            (default None; supported formats: .png, .tiff)).
        verbose: print more information? (default True).

    Returns:
    --------------------------------------------------------------------------
        regZ: the remapped TIN data.
    '''
    # Error handling:
    if not isinstance(input_topo, str):
        raise TypeError("'input_topo' must be of type 'str'")
    if not input_topo.endswith('.csv'):
        input_topo = input_topo + '.csv'
    if isinstance(savefig, str):
        if not (savefig.endswith('.png') or savefig.endswith('.tiff')):
            savefig = savefig + '.png'
    # Load the regular topographic grid:
    tmp = readRegularGrid(input_topo)
    rXY, nx, ny = tmp[:3]
    tmp = None
    # Load the Badlands output TIN:
    h5file = os.path.join(badlands_out, 'h5', 'tin.time{}.hdf5'.format(step))
    d = mapTIN2Reg(h5file, rXY, nx, ny)
    regZ = d['z']

    # Save the grid if asked:
    if isinstance(savegrid, str):
        save_topo_grid(
            savegrid, rXY[:, 0], rXY[:, 1], regZ.flatten(), verbose=verbose
        )
    # Save a figure if asked:
    if isinstance(savefig, str):
        fig = plt.figure(figsize=(10, 6), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(
            np.flipud(regZ),
            interpolation='nearest',
            cmap=cm.terrain,
            vmin=-4000,
            vmax=3000,
        )
        ax.contour(
            np.flipud(regZ), 0,
            colors='k',
            linewidths=1,
        )
        fig.subplots_adjust(right=0.95)
        fig.colorbar(im)
        plt.show(block=True)
        if verbose:
            print('Saving figure to {}... '.format(savefig),
                  end='', flush=True)
        fig.savefig(savefig)
        if verbose:
            print('Done')
        plt.close(fig)

    if return_all:
        return d
    return regZ


def readRegularGrid(input_topo):
    '''
    Read a regular grid and return some information.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        input_topo

    Returns:
    --------------------------------------------------------------------------
        rXY
        nx
        ny
        rZ
        dx
    '''
    rxyz = pd.read_csv(
        input_topo,
        sep=r'\s+',
        engine='c',
        header=None,
        na_filter=False,
        dtype=np.float,
        low_memory=False,
    )

    rX = rxyz.values[:, 0]
    rY = rxyz.values[:, 1]
    dx = rX[1] - rX[0]
    rXY = np.stack((rX, rY)).T
    nx, ny = get_nxy(rXY)

    rZ = np.reshape(rxyz.values[:, 2], (nx, ny), order='F').T

    return rXY, nx, ny, rZ, dx


def mapTIN2Reg(h5file, rXY, nx, ny):
    '''
    Read Badlands output, map to a regular grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        h5file
        rXY
        nx
        ny

    Returns:
    --------------------------------------------------------------------------
        data
    '''
    if not h5file.endswith('.hdf5'):
        h5file = h5file + '.hdf5'
    d = {}
    with h5py.File(h5file, 'r') as df:
        for key in df.keys():
            array = np.array((df['/' + key]))
            d[key] = array

    coords = d['coords']
    # Build cKDTree
    tree = cKDTree(coords[:, :2], leafsize=10)  # use x and y but not z
    dist, inds = tree.query(rXY, k=3)
    # Inverse weighting distance
    wght = 1.0 / dist ** 2
    onIDs = np.where(dist[:, 0] == 0)[0]

    # Iterate through keys in d
    data = {}
    for key in d:
        if key == 'connect':
            continue
        if key == 'coords':
            tinZ = d[key][:, 2]
            if tinZ[inds].ndim == 2:
                elev = (
                    np.sum(wght * tinZ[inds][:, :], axis=1)
                    / np.sum(wght, axis=1)
                )
            else:
                elev = (
                    np.sum(wght * tinZ[inds][:, :, 0], axis=1)
                    / np.sum(wght, axis=1)
                )
            if len(onIDs) > 0:
                elev[onIDs] = tinZ[inds[onIDs, 0]]
            elev = np.reshape(elev, (nx, ny), order='F').T
            data['z'] = elev
        else:
            array = d[key]
            if array[inds].ndim == 2:
                array = (
                    np.sum(wght * array[inds][:, :], axis=1)
                    / np.sum(wght, axis=1)
                )
            else:
                array = (
                    np.sum(wght * array[inds][:, :, 0], axis=1)
                    / np.sum(wght, axis=1)
                )
            if len(onIDs) > 0:
                array[onIDs] = array[inds[onIDs, 0]]
            array_new = np.reshape(array, (nx, ny), order='F').T
            data[key] = array_new

    return data


def get_nxy(rXY):
    '''
    Get dimensions of the regular grid from rXY.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        rXY

    Returns:
    --------------------------------------------------------------------------
        nx
        ny
    '''
    rX = rXY[:, 0]
    rY = rXY[:, 1]
    dx = rX[1] - rX[0]
    nx = int((rX.max()-rX.min())/dx + 1)
    ny = int((rY.max()-rY.min())/dx + 1)
    return nx, ny


def index_to_coords(index, rXY, nx, ny, order='ij'):
    '''
    Convert regular grid coordinates to projected model coordinates.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        index: (row, column) format
        rXY
        nx
        ny
        order: 'ij' or 'xy'
            (default 'ij')

    Returns:
    --------------------------------------------------------------------------
        x
        y
    '''
    if order == 'ij':
        i, j = index
    elif order == 'xy':
        j, i = index
    else:
        raise ValueError("Invalid 'order' parameter: {}".format(order))
    X = rXY[:, 0].reshape((ny, nx))
    Y = rXY[:, 1].reshape((ny, nx))
    x = X[i, j]
    y = Y[i, j]
    return x, y


def save_topo_grid(filename, X, Y, Z, verbose=False, **kwargs):
    '''
    Save the regular paleotopography grid.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        filename: output filename (.csv format)
        X, Y, Z: coordinates of the grid nodes
        verbose: if True, print more information
            (default False)
        **kwargs: any further kwargs are passed to pandas.DataFrame.to_csv
    '''
    if not filename.endswith('.csv'):
        filename += '.csv'
    if verbose:
        print(
            'Saving grid file to {}... '.format(filename), end='', flush=True
        )
    d = {'X': X, 'Y': Y, 'Z': Z.flatten()}
    df = pd.DataFrame(d)
    kw = {
        'columns': ['X', 'Y', 'Z'],
        'sep': ' ',
        'index': False,
        'header': False,
        'encoding': ENCODING,
    }
    for key in kw:
        if key in kwargs:
            kwargs.pop(key)
    df.to_csv(
        filename,
        **kw,
        **kwargs,
    )
    if verbose:
        print('Done')


def get_points_in_polygon(
        polygon_filenames,
        regular_grid,
        return_mask=False,
        return_regular_grid=False
):
    '''
    Return the indices of the points located within the polygon
    defined in polygon_filename.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        polygon_filename: the name of the .csv file containing the points
            extracted using ParaView, or an iterable of .csv filenames
        regular_grid: the .csv file containing the base for the regular grid
            (usually the present-day topography .csv file)

    Returns:
    --------------------------------------------------------------------------
        fillIDs: the indices of the points located within the polygon
    '''
    if isinstance(polygon_filenames, str):
        polygon_filenames = [polygon_filenames]
    if not regular_grid.endswith('.csv'):
        regular_grid += '.csv'

    rXY, nx, ny, rZ, dx = readRegularGrid(regular_grid)

    Xms = []
    Yms = []
    for polygon_filename in polygon_filenames:
        if not polygon_filename.endswith('.csv'):
            polygon_filename += '.csv'
        xyz = pd.read_csv(
            polygon_filename,
            sep=',',
            engine='c',
            na_filter=False,
            dtype=np.float,
            low_memory=False,
        )
        Xm = xyz['Points:0'].values[:]
        Xms.append(Xm.reshape((-1, 1)))
        Ym = xyz['Points:1'].values[:]
        Yms.append(Ym.reshape((-1, 1)))

    Xm = np.vstack(Xms)
    Ym = np.vstack(Yms)

    XYm = np.hstack((Xm, Ym))
    mtree = cKDTree(XYm, leafsize=10)
    distances = mtree.query(rXY, k=1)[0]
    onIDs = np.where(distances[:] <= 1.5 * dx)[0]
    in_polygon = np.zeros(len(rZ.flatten()))
    in_polygon[onIDs] = 1
    in_polygon = np.reshape(in_polygon, (nx, ny), order='F').T
    in_polygon = gaussian_filter(in_polygon, sigma=1)
    mask = in_polygon > 0.1
    fillIDs = np.where(mask)

    output = [fillIDs]
    if return_mask:
        output.append(mask)
    if return_regular_grid:
        output.extend([rXY, nx, ny, rZ, dx])
    return tuple(output)
