import json

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# from osgeo import gdal
# from skimage import transform


# def extract_etopo(filename, llc, urc, shape=None):
#     raster = gdal.Open(filename)
#     Z = raster.ReadAsArray()
#     if filename.endswith('.tif') or filename.endswith('.tiff'):
#         Z = np.flipud(Z)

#     lons = np.linspace(-180, 180, Z.shape[1])
#     lats = np.linspace(-90, 90, Z.shape[0])
#     dx = lons[1] - lons[0]
#     dy = lats[1] - lats[0]

#     X, Y = np.meshgrid(lons, lats)

#     lat_min, lon_min = llc
#     lat_max, lon_max = urc

#     n_lons = int(np.around((lon_max - lon_min) / dx, 0)) + 1
#     n_lats = int(np.around((lat_max - lat_min) / dy, 0)) + 1

#     mask = np.logical_and.reduce((
#         X >= lon_min,
#         X <= lon_max,
#         Y >= lat_min,
#         Y <= lat_max,
#     ))

#     Z = Z[mask].reshape((n_lats, n_lons))

#     if shape is not None:
#         Z = transform.resize(
#             Z, shape, anti_aliasing=True, preserve_range=True, order=3
#         )

#     return Z


json_dump = """
[
    {
        "ColorSpace" : "RGB",
        "Name" : "Preset",
        "Points" :
        [
            -11455.0,
            0.0,
            0.5,
            0.0,
            6000.0,
            1.0,
            0.5,
            0.0
        ],
        "RGBPoints" :
        [
            -11455.0,
            0.141176,
            0.14902000000000001,
            0.68627499999999997,
            -5807.8017540748197,
            0.219608,
            0.22745099999999999,
            0.764706,
            -5807.8017540748197,
            0.219608,
            0.22745099999999999,
            0.764706,
            -3240.8852900287802,
            0.27450999999999998,
            0.28235300000000002,
            0.83921599999999996,
            -3240.8852900287802,
            0.27450999999999998,
            0.28235300000000002,
            0.83921599999999996,
            -2214.1246960690601,
            0.31764700000000001,
            0.40000000000000002,
            0.85097999999999996,
            -2214.1246960690601,
            0.31764700000000001,
            0.40000000000000002,
            0.85097999999999996,
            -930.66446682647404,
            0.39215699999999998,
            0.50588200000000005,
            0.87451000000000001,
            -930.66346821669094,
            0.39215699999999998,
            0.50588200000000005,
            0.87451000000000001,
            -540.49564084374094,
            0.51372499999999999,
            0.63137299999999996,
            0.90196100000000001,
            8.3412951319551194,
            1.0,
            1.0,
            1.0,
            9.3399049160998402,
            0.062745099999999998,
            0.478431,
            0.18431400000000001,
            9.3399049160998402,
            0.54901999999999995,
            0.72548999999999997,
            0.83921599999999996,
            9.3399049160998402,
            0.0,
            0.38039200000000001,
            0.27843099999999998,
            9.3399049160998402,
            0.66666700000000001,
            0.78431399999999996,
            1.0,
            22.824032970427702,
            0.38039200000000001,
            0.61568599999999996,
            0.298039,
            188.50647794819801,
            0.062745099999999998,
            0.478431,
            0.18431400000000001,
            221.642347805691,
            0.494118,
            0.66666700000000001,
            0.34117599999999998,
            352.79176797697801,
            0.90980399999999995,
            0.84313700000000003,
            0.49019600000000002,
            352.79276658676298,
            0.90980399999999995,
            0.84313700000000003,
            0.49019600000000002,
            1071.5311740172599,
            0.63137299999999996,
            0.26274500000000001,
            0.0,
            1071.5311740172599,
            0.63137299999999996,
            0.26274500000000001,
            0.0,
            1584.9064779482001,
            0.50980400000000003,
            0.117647,
            0.117647,
            1584.9064779482001,
            0.50980400000000003,
            0.117647,
            0.117647,
            2714.3541160115101,
            0.43137300000000001,
            0.43137300000000001,
            0.43137300000000001,
            2714.3541160115101,
            0.43137300000000001,
            0.43137300000000001,
            0.43137300000000001,
            3946.4688259827299,
            1.0,
            1.0,
            1.0,
            3946.4688259827299,
            1.0,
            1.0,
            1.0,
            6000.0,
            1.0,
            1.0,
            1.0
        ]
    }
]
"""


def _func(json_dump):
    dump = json.loads(json_dump)

    points_list = dump[0]['RGBPoints']
    points_list = [float(i) for i in points_list]
    elevations = points_list[::4]
    elevations = np.asarray(elevations)
    elevations -= elevations.min()
    elevations /= elevations.max()

    colours = points_list[:]
    del colours[0::4]
    colours = [tuple(colours[i:i + 3]) for i in range(0, len(colours), 3)]
    d = {i: j for i, j in zip(elevations, colours)}

    colours_list = []
    for key in d:
        value = d[key]
        colours_list.append((key, value))
    etopo_cmap = LinearSegmentedColormap.from_list('etopo', colours_list)

    return etopo_cmap, colours_list


etopo_cmap, colours_list = _func(json_dump)

etopo_min = -11455.0
etopo_max = 6000.0

etopo_kwargs = {
    'vmin': etopo_min,
    'vmax': etopo_max,
    'cmap': etopo_cmap,
}
