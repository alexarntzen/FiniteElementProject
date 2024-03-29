import numpy as np

# quadrature weights
quadrature_table_1D = {
    1: [[0, 2]],
    2: [[- 1 / np.sqrt(3), 1],
        [1 / np.sqrt(3), 1]],
    3: [[- np.sqrt(3 / 5), 5 / 9],
        [0, 8 / 9],
        [np.sqrt(3 / 5), 5 / 9]],
    4: [[- np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36],
        [- np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
        [+ np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
        [+ np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36]]
}

# quadrature weights
quadrature_table_2D = {
    1: [[(1 / 3, 1 / 3, 1 / 3), 1]],
    3: [[(1 / 2, 1 / 2, 0), 1 / 3],
        [(1 / 2, 0, 1 / 2), 1 / 3],
        [(0, 1 / 2, 1 / 2), 1 / 3]],
    4: [[(1 / 3, 1 / 3, 1 / 3), -9 / 16],
        [(3 / 5, 1 / 5, 1 / 5), 25 / 48],
        [(1 / 5, 3 / 5, 1 / 5), 25 / 48],
        [(1 / 5, 1 / 5, 3 / 5), 25 / 48]]
}


# approximates the integral form a to b of g with Nq evaluations
def quadrature1D(a, b, Nq, g):
    I = 0
    for row in quadrature_table_1D[Nq]:
        I += g(row[0] * (b - a) / 2 + (b + a) / 2) * row[1]
    return I * np.linalg.norm(b - a) / 2



# return the function that maps barycentric coordinates to R2
def make_barycentric_to_R2(p1, p2, p3):
    return lambda z1, z2, z3: z1 * p1 + z2 * p2 + z3 * p3


# Get the area of the triangle with vertices p1, p1 and p3
def vertices_to_area_2D(p1, p2, p3):
    return np.abs(np.linalg.det([p1 - p3, p2 - p3]) / 2)


# approximates the integral in the triangle p1, p2 p3 of g with Nq evaluations
def quadrature2D(p1, p2, p3, Nq, g):
    barycentric_to_R2 = make_barycentric_to_R2(p1, p2, p3)
    I = 0
    for row in quadrature_table_2D[Nq]:
        I += g(barycentric_to_R2(*(row[0]))) * row[1]
    return I * vertices_to_area_2D(p1, p2, p3)
