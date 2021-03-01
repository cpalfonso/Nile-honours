from matplotlib.path import Path
import numpy as np


def point_in_polygon(point_coords, polygon_coords):
    if point_coords.shape[0] == 2:
        point_coords = point_coords.T
    elif point_coords.shape[1] != 2:
        raise ValueError(
            "'point_coords' must be of shape (N, 2) "
            + "or (2, N)"
        )
    if polygon_coords.shape[0] == 2:
        polygon_coords = polygon_coords.T
    elif polygon_coords.shape[1] != 2:
        raise ValueError(
            "'polygon_coords' must be of shape (N, 2) "
            + "or (2, N)"
        )

    point_x = point_coords[:, 0]
    point_y = point_coords[:, 1]
    polygon_x = polygon_coords[:, 0]
    polygon_y = polygon_coords[:, 1]
    x0 = polygon_x.min()
    x1 = polygon_x.max()
    y0 = polygon_y.min()
    y1 = polygon_y.max()

    inds = np.where(np.logical_and.reduce((
        point_x >= x0,
        point_x <= x1,
        point_y >= y0,
        point_y <= y1,
    )))[0]
    path = Path(polygon_coords)
    inside = np.zeros(point_coords.shape[0]).astype(bool)
    points_tmp = np.hstack((
        point_x[inds].reshape((-1, 1)),
        point_y[inds].reshape((-1, 1)),
    ))
    inside_tmp = path.contains_points(points_tmp)
    inside[inds[inside_tmp]] = True
    return inside
