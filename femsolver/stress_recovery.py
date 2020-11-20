import numpy as np


def get_derivative(U, p, element):
    # get the derrivative of an element given displacement U
    # This is not the most effective approximation but it is so beautiful
    p1, p2, p3 = p[element[0]], p[element[1]], p[element[2]]
    v1 = p2 - p1
    v2 = p3 - p1
    V = np.array([v1, v2])
    # Derrivative in v coordinate system
    DU_V = np.array([
        (U[element[1]] - U[element[0]]),
        (U[element[2]] - U[element[0]])
    ])

    # Derivative in standard basis
    return np.linalg.inv(V) @ DU_V


def get_strain_vector(U, p, element):
    # return the strain given by the derivative
    DU = get_derivative(U, p, element)
    return np.array([
        DU[0, 0],
        DU[1, 1],
        DU[0, 1] + DU[1, 0]])


def get_stress(U, p, element, C):
    # get stress from strain vector
    strain_vector = get_strain_vector(U, p, element)
    stress_vector = C @ strain_vector
    # return strain tensor
    return np.array([
        [stress_vector[0], stress_vector[2]],
        [stress_vector[2], stress_vector[1]]
    ])


def get_naive_stress_recovery(U, p, tri, C):
    nodes, k = U.shape
    element_per_node = np.zeros(nodes, dtype=np.uint64)
    stress_recovery = np.zeros((nodes, k, k))
    for element in tri:
        # Find the stress of each element
        sigma = get_stress(U, p, element, C)
        for index in element:
            # add the stress to all the nodes in the element
            stress_recovery[index] += sigma
            element_per_node[index] += 1
    # Averaging over number of patches neighbouring the nodes
    for i in range(nodes):
        stress_recovery[i] /= element_per_node[i]
    return stress_recovery
