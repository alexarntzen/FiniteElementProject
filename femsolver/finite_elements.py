import numpy as np

class LinearTriangle:
    ref_element = np.array([[1, 0], [0, 1], [0, 0]])
    shape_fun = lambda ksi : np.array([[1, 0, -1], [0, 1, -1]]).T @ ksi + np.array([0, 0, 1])
    shape_fun_jacobian = lambda ksi : np.array([[1, 0, -1], [0, 1, -1]]).T

class IsogeometricLinearTriangle:
    displacement = LinearTriangle
    geometry = LinearTriangle