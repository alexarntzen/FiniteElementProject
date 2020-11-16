import numpy as np

import femsolver.quadrature as qd
from femsolver.finite_elements import IsogeometricLinearTriangle


def compose(f, g):
    return lambda x: f(g(x))


def function_multiply(f, g):
    return lambda x: f(x) * g(x)


# Warning: meshing and plotting only supports the linear triangle
# But this solver can use any valid shape functions in 2D
def get_A_F(p, tri, dirichlet_edges, f, g=None, neumann_edges=np.empty(0), Nq=4,
            finite_element=IsogeometricLinearTriangle, tri_u=None):
    # find the shape functions for the reference triangle
    if tri_u is None:
        tri_u = tri
    ref_element_geom = finite_element.geometry.ref_element
    sf_geom = finite_element.geometry.shape_fun
    sf_geom_jac = finite_element.geometry.shape_fun_jacobian
    sf_u = finite_element.displacement.shape_fun
    sf_u_jac = finite_element.displacement.shape_fun_jacobian
    n_bar = len(np.unique(tri_u))
    A = np.zeros((n_bar, n_bar))
    F = np.zeros(n_bar)

    for element, element_u in zip(tri, tri_u):
        # find expression for the dual basis for the reference element
        X = p[element].T

        def left_integrand(ksi):
            left = sf_u_jac(ksi) @ np.linalg.inv(X @ sf_geom_jac(ksi))
            jacobian_det = np.linalg.det(X @ sf_geom_jac(ksi))
            return left @ left.T * jacobian_det

        def right_integrand(ksi):
            jacobian_det = np.linalg.det(X @ sf_geom_jac(ksi))
            return f(X @ sf_geom(ksi)) * sf_u(ksi) * jacobian_det

        # finding A matrix
        # A[element[alpha], element[beta]] += A_i[alpha,beta]
        # For linear geometry and solution shape functions we can calculate this integral only once, then scale it.
        A[np.ix_(element_u, element_u)] += qd.quadrature2D(*ref_element_geom, Nq=1, g=left_integrand)

        # Add load to F vector
        F[element_u] += qd.quadrature2D(*ref_element_geom, Nq, right_integrand)

        # apply neumann conditions if applicable
        for alpha in range(len(element)):
            for beta in range(len(element)):
                if [element[alpha], element[beta]] in neumann_edges.tolist():
                    vertex1, vertex2 = ref_element_geom[[alpha, beta]]

                    def right_integrand_neumann(ksi):
                        coor_change = np.linalg.norm(X @ sf_geom_jac(ksi) @ (vertex2 - vertex1)
                                                     ) / np.linalg.norm(vertex1 - vertex2)
                        return sf_u(ksi) * g(X @ sf_geom(ksi)) * coor_change

                    F[element_u] += qd.quadrature1D(vertex1, vertex2, Nq, g=right_integrand_neumann)

                    # F[element[beta]] += qd.quadrature1D(vertex1, vertex2, Nq, function_multiply(H, g))

    # Applying dirichlet boundary conditions
    epsilon = 1e-100
    dirichlet_vertecis = np.unique(dirichlet_edges)
    for i in dirichlet_vertecis:
        A[i, i] = 1 / epsilon
        F[i] = 0

    return A, F


def solve(p, tri, dirichlet_edges, f, g=None, neumann_edges=np.empty(0), Nq=4):
    A, F = get_A_F(p, tri, dirichlet_edges, f, g, neumann_edges, Nq,
            finite_element=IsogeometricLinearTriangle, tri_u=None )
    U = np.linalg.solve(A, F)
    return U
