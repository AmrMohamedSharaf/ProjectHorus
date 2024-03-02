import numpy as np
from scipy.spatial import KDTree
from utils import transform_to_polynomial_basis




#TODO lets have a class constructer with let's say construction method , and it takes point data as an input 


# Moving least squares method 

# Step one Implicit representation  : 

# Constraitns points 
def sample_constraints(vertices, normals, eps):
    # getting the shape of the vertex arrays 
    n_points = vertices.shape[0]
    new_vert_pos, new_values_pos = np.zeros((n_points , vertices.shape[1])), np.zeros(n_points)
    new_vert, new_values = np.zeros((n_points*2 , vertices.shape[1])), np.zeros(n_points * 2)
    kdtree = KDTree(vertices) # nearest point search 
    new_vertices_pos = vertices + eps * normals #offset vectors for the first itr
    new_values_pos [:n_points] = eps
    # first loop for the positive offset 
    while True:
        # closest points querey 
        _,closest_point_indices_pos = kdtree.query(new_vertices_pos)
        violation_mask = closest_point_indices_pos != np.arange(n_points) # the corresponding ones should show a one to one relationship with verticees 
        if not np.any(violation_mask):
            break  # the mask doesn't catch any violationtns then exsit 
        new_vertices_pos[violation_mask] = vertices[violation_mask]+normals[violation_mask]*eps
        # if not then try for a lower eps value for all sampling points that were caught using the mask 
        new_vert_pos[np.where(violation_mask)]
        new_values_pos[violation_mask] = eps
        
        eps /= 2.0

    # same procedure but for negative offsets 
    new_vert_neg, new_values_neg = np.zeros((n_points , vertices.shape[1])), np.zeros(n_points)
    new_vert, new_values = np.zeros((n_points*2 , vertices.shape[1])), np.zeros(n_points * 2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    kdtree = KDTree(vertices)
    new_vert_neg = vertices - eps * normals
    new_values_neg [:n_points] = -eps
    
    while True:
        _,closest_point_indices_neg = kdtree.query(new_vert_neg)
        violation_mask = closest_point_indices_neg != np.arange(n_points)
        if not np.any(violation_mask):
            break  
        new_vert_neg[violation_mask] = vertices[violation_mask]-normals[violation_mask]*eps

        new_values_neg[violation_mask] = -eps
        
        eps /= 2.0

    new_vert[:n_points] = new_vertices_pos
    new_vert[n_points:] = new_vert_neg
    new_values[:n_points] = new_values_pos
    new_values[n_points:] = new_values_neg
    return new_vert, new_values




def global_predictor(grid_pts, constr_pts, constr_vals, degree=2):
    """Evaluate implicit function in space

    Args:
        grid_pts (np.array [N, 3]): 3D coordinates of N points in space. Grid points
        constr_pts (np.array [M, 3]): 3D coordinates of M points in space. Constraints points
        constr_vals (np.array, [M, 1]): constraint values defined on constr_pts
        degree (int): degree of Polynomial

    Returns:
        pred_vals (np.array [N, 1]): implicit function values for each of the grid points
    """
    pred_vals = np.zeros((len(grid_pts), 1))

    nearby_vertices_poly = transform_to_polynomial_basis(constr_pts, degree)
    grid_pt_poly = transform_to_polynomial_basis(grid_pts, degree)
    coefficients = np.linalg.solve(nearby_vertices_poly.T @ nearby_vertices_poly, nearby_vertices_poly.T @ constr_vals)
    pred_vals = grid_pt_poly @ coefficients









