'''
Class to be used for constructing Badlands input grids.
'''
# TODO: Change GridBuilder attributes to properties where appropriate.

import numpy as np
from pyproj import Proj
from scipy.ndimage import map_coordinates

from ..io import nc, xy
# from .. import pygplates_tools as pgpt


class GridBuilder:
    '''
    Class used for constructing Badlands input grids.

    --------------------------------------------------------------------------
    Attributes and methods:
        __init__(llc, urc, resolution, projection, order='lat_lon'): create
            a GridBuilder instance using all of the necessary information.
        lats: (N,) array of latitudes of grid nodes.
        lons: (N,) array of longitudes of grid nodes.
        nx: width of the grid.
        ny: height of the grid.
        x: (N,) array of projected x-coordinates of grid nodes.
        y: (N,) array of projected y-coordinates of grid nodes.
        z: if map_etopo has been called, (ny, nx) array of elevations of
            grid nodes; else None.
        get_lon_lat(): get an (N, 2) array of the (lon, lat) coordinate
            pairs which make up the grid.
        map_climate(polygons_filename): map a precipitation .xy file to
            the grid.
        map_etopo(etopo_filename, order=3, **kwargs): map a global or
            regional ETOPO1 netCDF4 file to the grid.
        map_nc(nc_filename, order=3, **kwargs): map a global or regional
            netCDF4 file to the grid, using scipy.ndimage.map_coordinates.
        map_tectonics(polygons_filename): map a tectonics .xy file to the
            grid.
        points_in_polygon(polygon, check_bounds=True): wrapper for
            paleoflow.pygplates_tools.points_in_polygon.
    '''
    def __init__(
        self,
        llc,
        urc,
        resolution,
        projection,
        order='lat_lon',
    ):
        '''
        Create a GridBuilder instance using all the necessary information.

        ----------------------------------------------------------------------
        Parameters:
            llc: (lat, lon) or (lon, lat) coordinate tuple representing the
                lower-left corner of the grid.
            urc: (lat, lon) or (lon, lat) coordinate tuple representing the
                upper-right corner of the grid.
            resolution: resolution (in metres) of the grid.
            projection: pyproj.Proj object, or a Proj4 string, representing
                the projection to be used by the grid.
            order (optional): the order of the coordinate tuples llc and urc.
                Valid options are 'lat_lon', 'lon_lat', 'yx', 'xy', 'ij',
                'ji', 'latlon', 'lonlat', 'lat-lon', 'lon-lat'.
                (default 'lat_lon')
        Returns:
            gb: GridBuilder instance.
        '''
        if order in ['lat_lon', 'yx', 'ij', 'latlon', 'lat-lon']:
            order = 'lat_lon'
        elif order in ['lon_lat', 'xy', 'ji', 'lonlat', 'lon-lat']:
            order = 'lon_lat'
        else:
            raise ValueError(
                "Unrecognised 'order' parameter: {}".format(order)
            )
        if order == 'lat_lon':
            llc = llc[::-1]
            urc = urc[::-1]
        if not isinstance(projection, Proj):
            projection = Proj(projection)
        # Get extents
        proj_x0, proj_y0 = projection(*llc)
        proj_x1, proj_y1 = projection(*urc)
        # Build grid
        xcoords = np.arange(
            proj_x0, proj_x1, resolution,
        )
        ycoords = np.arange(
            proj_y0, proj_y1, resolution,
        )
        X, Y = np.meshgrid(xcoords, ycoords)
        self.nx = X.shape[1]
        self.ny = X.shape[0]

        self.x = X.flatten()
        self.y = Y.flatten()
        self.z = None

        self.lons, self.lats = projection(
            self.x, self.y, inverse=True,
        )

    def map_etopo(self, etopo_filename, order=3, **kwargs):
        '''
        Map a global or regional ETOPO1 netCDF4 file to the grid, using
        scipy.ndimage.map_coordinates and GridBuilder.map_nc. The result
        is returned, and also placed in self.z.

        ----------------------------------------------------------------------
        Parameters:
            nc_filename: path to the ETOPO1 netCDF4 file to be mapped.
            order (optional): order of the interpolation to be used.
                (default 3)
            kwargs (optional): any further keyword arguments are passed
                to scipy.ndimage.map_coordinates.
        Returns:
            mapped: the data from the netCDF4 file, mapped to the grid.
                Array dimensions are (self.ny, self.nx).
        '''
        zcoords = self.map_nc(etopo_filename, order=order, **kwargs)
        self.z = zcoords
        return zcoords

    def map_nc(self, nc_filename, order=3, **kwargs):
        '''
        Map a global or regional netCDF4 file to the grid, using
        scipy.ndimage.map_coordinates.

        ----------------------------------------------------------------------
        Parameters:
            nc_filename: path to the netCDF4 file to be mapped.
            order (optional): order of the interpolation to be used.
                (default 3)
            kwargs (optional): any further keyword arguments are passed
                to scipy.ndimage.map_coordinates.
        Returns:
            mapped: the data from the netCDF4 file, mapped to the grid.
                Array dimensions are (self.ny, self.nx).
        '''
        lon, lat, z = nc.read(nc_filename)
        lon, lat = np.meshgrid(lon, lat)
        dx = np.abs(lon[0, 1] - lon[0, 0])
        dy = np.abs(lat[1, 0] - lat[0, 0])
        xcoords = (self.lons - lon.min()) / dx
        ycoords = (self.lats - lat.min()) / dy
        xycoords = np.vstack((
            ycoords.reshape((1, -1)),
            xcoords.reshape((1, -1)),
        ))
        mapped = map_coordinates(
            z,
            xycoords,
            order=order,
            **kwargs,
        ).reshape((self.ny, self.nx))
        return mapped

    def map_tectonics(self, polygons_filename):
        '''
        Map a tectonics .xy file to the grid.

        ----------------------------------------------------------------------
        Parameters:
            polygons_filename: path to the .xy file containing the tectonic
                polygons.
        Returns:
            tectonics: Array (shape (self.ny, self.nx)) containing the mapped
                tectonic data.
        '''
        tectonics = np.zeros((self.ny, self.nx)).flatten()
        features = xy.read(polygons_filename, True)
        if features is None:
            return tectonics.reshape((self.ny, self.nx))
        for feature in features:
            polygon = feature['geometry']
            name = feature['name']
            value = float(name.split('_')[-2])
            inside = self.points_in_polygon(polygon)
            tectonics[inside] = value
        return tectonics.reshape((self.ny, self.nx))

    def map_climate(self, polygons_filename):
        '''
        Map a precipitation .xy file to the grid.

        ----------------------------------------------------------------------
        Parameters:
            polygons_filename: path to the .xy file containing the climate
                polygons.
        Returns:
            precipitation: Array (shape (self.ny, self.nx)) containing the
                mapped precipitation data.
        '''
        precipitation = np.zeros((self.ny, self.nx)).flatten()
        features = xy.read(polygons_filename, True)
        if features is None:
            return precipitation.reshape((self.ny, self.nx))
        for feature in features:
            polygon = feature['geometry']
            name = feature['name']
            value = float(name.split('_')[-1])
            inside = self.points_in_polygon(polygon)
            precipitation[inside] = value
        return precipitation.reshape((self.ny, self.nx))

    def get_lon_lat(self):
        '''
        Get an (N, 2) array of the (lon, lat) coordinate pairs which make
        up the grid.

        ----------------------------------------------------------------------
        Returns:
            lon_lat: an array with dimensions (self.nx * self.ny, 2),
                containing the (lon, lat) coordinate pairs.
        '''
        lon_lat = np.hstack((
            self.lons.reshape((-1, 1)),
            self.lats.reshape((-1, 1)),
        ))
        return lon_lat

    def get_x(self):
        '''
        Return an array containing the unique x-values of the grid nodes.

        ----------------------------------------------------------------------
        Returns:
            x: a 1-D array (length self.nx) containing the x-values.
        '''
        return self.x.reshape((self.ny, self.nx))[int(self.ny / 2), :]

    def get_y(self):
        '''
        Return an array containing the unique y-values of the grid nodes.

        ----------------------------------------------------------------------
        Returns:
            y: a 1-D array (length self.ny) containing the y-values.
        '''
        return self.y.reshape((self.ny, self.nx))[:, int(self.nx / 2)]

    # def points_in_polygon(self, polygon, check_bounds=True):
    #     '''
    #     Wrapper for paleoflow.pygplates_tools.points_in_polygon.

    #     ----------------------------------------------------------------------
    #     Parameters:
    #         polygon: (N, 2) array of (lon, lat) coordinate pairs.
    #         check_bounds: if True, check the x-y bounding box of
    #             the polygon first.
    #             (default True)
    #     Returns:
    #         result: (N,) boolean array.
    #     '''
    #     lon_lat = self.get_lon_lat()
    #     result = pgpt.points_in_polygon(
    #         polygon, lon_lat, check_bounds=check_bounds,
    #     )
    #     return result
