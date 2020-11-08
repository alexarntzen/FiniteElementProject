import numpy as np


def get_derivative(U, p, element):
    # This is not the most effective approximation but it is so beautiful
    p1, p2, p3 = p[element[0]], p[element[1]], p[element[2]]
    v1 = p2 - p1
    v2 = p3 - p1
    V = np.array([v1,v2])
    DU_V = np.array([
        (U[element[1]] - U[element[0]]),
        (U[element[2]] - U[element[0]])
    ])
    return np.linalg.inv(V)@DU_V


def get_strain(U, p, tri):
    return 1


def get_eps(U, p, tri):
    return 1


def get_naive_stress_recovery(U, p, tri):
    return 1
