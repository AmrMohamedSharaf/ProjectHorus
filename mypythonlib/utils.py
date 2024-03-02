import numpy as np
from scipy.spatial import KDTree

def transform_to_polynomial_basis(pts, degree):
    """ Represent 2D/3D points (pts) in polynomial basis of given degree
    e.g. degree 2: (x, y) -> (1, x, y, x^2, y^2, xy)
    
    Args:
        pts (np.array [N, 2 or 3]): 2D/3D coordinates of N points in space
        degree (int): degree of Polynomial

    Returns:
        ext_pts (np.array [N, ?]): points (pts) in Polynomial basis of given degree. 
        The second dimension depends on the polynomial degree and initial dimention of points (2D or 3D)  
    """
    if degree == 0:
        return np.ones([len(pts), 1])
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    ext_pts = np.concatenate([np.ones([len(pts), 1]), x, y], axis=1)
    for i in range(2, degree + 1):
        for j in range(i + 1):
            term = (x ** (i - j)) * (y ** j)
            ext_pts = np.concatenate([ext_pts, term], axis=1)
    return ext_pts