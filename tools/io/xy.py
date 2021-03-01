'''
Functions to read .xy files.
'''

import numpy as np
import pandas as pd


class _XYReader:
    def __init__(self, filename=None):
        self.filename = str(filename)

    def is_empty(self):
        if self.filename is None:
            return True
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        var = 0
        for i in range(len(lines) - 1):
            line = lines[i].rstrip()
            if not line.startswith('>'):
                var += 1
        if var == 0:
            return True
        return False

    def read(self, geometry_array=False):
        if self.is_empty():
            return None
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        features = []

        data = {}
        points = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('>') and len(points) > 0:
                data['geometry'] = points
                features.append(data)
                data = {}
                points = []
            if line.startswith('>') and not line.startswith('> '):
                continue
            if line.startswith('> '):
                key = line.split()[1]
                value = ' '.join(line.split()[2:])
                data[key] = value
            else:
                points.append((float(line.split()[0]), float(line.split()[1])))

        if geometry_array:
            for feature in features:
                geom_list = feature['geometry']
                geom_array = np.zeros((len(geom_list), 2))
                for i, _ in enumerate(geom_list):
                    geom_array[i, 0] = geom_list[i][0]
                    geom_array[i, 1] = geom_list[i][1]
                feature['geometry'] = geom_array

        return features

    def read_velocities(self):
        if is_empty(self.filename):
            return None
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        features = []

        data = {}
        values = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('>') and len(values) > 0:
                data['data'] = values
                features.append(data)
                data = {}
                values = []
            if line.startswith('>') and not line.startswith('> '):
                continue
            if line.startswith('> '):
                key = line.split()[1]
                value = ' '.join(line.split()[2:])
                data[key] = value
            else:
                values.append(
                    [
                        float(i) for i in line.split()
                    ][:-1]
                )

        # Features is length 1 list
        # Features[0] is a dictionary
        # Features[0]['data'] is a list of lists
        feature = features[0]
        data = feature['data']
        arr = np.zeros((len(data), len(data[0])))
        for i, _ in enumerate(data):
            for j, _ in enumerate(data[i]):
                arr[i, j] = float(data[i][j])
        df = pd.DataFrame({
            'lon': arr[:, 0],
            'lat': arr[:, 1],
            'vx': arr[:, 3],
            'vy': -1 * arr[:, 2],
        })
        return df


def is_empty(filename):
    reader = _XYReader(filename)
    return reader.is_empty()


def read(filename, geometry_array=False):
    reader = _XYReader(filename)
    return reader.read(geometry_array=geometry_array)


def read_velocities(filename):
    '''
    Read a (lon, lat) format velocity .xy file exported from GPlates.

    --------------------------------------------------------------------------
    Parameters:
        filename: path to the velocity .xy file.
    Returns:
        df: pandas.DataFrame containing the velocity data.
    '''
    reader = _XYReader(filename)
    return reader.read_velocities()


# Alias for read()
load = read
