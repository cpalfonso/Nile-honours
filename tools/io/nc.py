'''
Functions for simplifying reading and writing .nc files.
'''

import os

from netCDF4 import Dataset
import numpy as np


def read(
        filename,
        variables=('lon', 'lat', 'z'),
):
    '''
    Read a .nc file.
    ----------------------------------------------------
    Parameters:
        filename: path to the input file
        variables: the variables to read from the input file
            (default ('lon', 'lat', 'z'))
    Returns:
        vals: tuple of numpy.array objects corresponding to
            'variables'
    '''
    if not filename.endswith('.nc'):
        filename += '.nc'
    if isinstance(variables, str):
        variables = [variables]
    val = []
    with Dataset(filename, 'r', format='NETCDF4') as data:
        for i in variables:
            val.append(np.asarray(data[i]))
    if len(val) == 1:
        return val[0]
    else:
        return tuple(val)


# Alias for read()
load = read


def write(
        data,
        lons,
        lats,
        output_file,
        title=None,
        description=None,
        history=None,
        zlib=True,
        verbose=True,
):
    if not output_file.endswith('.nc'):
        output_file += '.nc'
    if not os.path.isdir(os.path.dirname(output_file)):
        output_dir = os.path.abspath(os.path.dirname(output_file))
        raise FileNotFoundError(
            "Output directory does not exist: {}".format(output_dir)
        )
    if os.path.isfile(output_file):
        if verbose:
            print("Output file '{}'".format(output_file)
                  + " already exists; overwriting...")
        os.remove(output_file)
    with Dataset(output_file, 'w', format='NETCDF4') as rootgrp:
        rootgrp.createDimension('lon', len(lons))
        rootgrp.createDimension('lat', len(lats))
        longitudes = rootgrp.createVariable(
            'lon', 'f8', ('lon',), zlib=zlib
        )
        longitudes[:] = lons
        latitudes = rootgrp.createVariable(
            'lat', 'f8', ('lat',), zlib=zlib
        )
        latitudes[:] = lats
        zs = rootgrp.createVariable(
            'z', 'f4', ('lat', 'lon'), zlib=zlib
        )
        zs[:] = data.reshape((len(lats), len(lons)))

        if title is not None:
            rootgrp.title = title
        if description is not None:
            rootgrp.description = description
        if history is not None:
            rootgrp.history = history

    if verbose:
        print('Output file {} created'.format(output_file))
