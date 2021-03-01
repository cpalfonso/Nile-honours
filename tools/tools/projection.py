import numpy as np
from pyproj import transform


def is_valid_lat_lon(coords):
    try:
        if len(coords) != 2:
            return False
    except TypeError:
        return False
    try:
        if coords[0] < -90 or coords[0] > 90:
            return False
        if coords[1] < -180 or coords[1] > 180:
            return False
    except TypeError:
        return False
    return True


def is_valid_lon_lat(coords):
    try:
        if len(coords) != 2:
            return False
    except TypeError:
        return False
    try:
        if coords[0] < -180 or coords[0] > 180:
            return False
        elif coords[1] < -90 or coords[1] > 90:
            return False
    except TypeError:
        return False
    return True


def test_valid_lat_lon(coords):
    if not is_valid_lat_lon(coords):
        raise ValueError("'{}' is not a valid (lat, lon) pair".format(coords))


def test_valid_lon_lat(coords):
    if not is_valid_lon_lat(coords):
        raise ValueError("'{}' is not a valid (lon, lat) pair".format(coords))


def array_project(
        d,
        input_proj,
        output_proj,
        order='lon_lat',
):
    '''
    Project an (n, 2) array of (lon, lat) or (lat, lon) coordinate
        pairs.
    Parameters:
        d: the input array, in (lon, lat) or (lat, lon) form.
        input_proj, output_proj: pyproj.Proj objects.
        order: 'lon_lat' or 'lat_lon', indicating (lon, lat) or
            (lat, lon) coordinate pairs in d, respectively.
    Returns:
        d_project: the projected coordinates from d, in (x, y) or
            (y, x) form, depending on 'order'.
    '''
    # Process 'order' parameter
    if order == 'lon_lat':
        lons = d[:, 0]
        lats = d[:, 1]
    elif order == 'lat_lon':
        lats = d[:, 0]
        lons = d[:, 1]
    else:
        raise ValueError("Invalid 'order' parameter: {}".format(order))
    fx, fy = transform(
        input_proj,
        output_proj,
        lons,
        lats,
    )
    if order == 'lon_lat':
        d_project = np.dstack([fx, fy])[0]
    else:
        d_project = np.dstack([fy, fx])[0]
    return d_project
