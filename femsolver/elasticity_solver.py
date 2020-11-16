import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import femsolver.quadrature as qd
from femsolver.finite_elements import IsogeometricLinearTriangle


def compose(f, g):
    return lambda x: f(g(x))


def function_multiply(f, g):
    return lambda x: f(x) * g(x)


def index(node, d):
    return 2 * (node) + d


def reshape_U(U):
    two_N = len(U)
    return U.reshape(two_N // 2, 2)


def proj(f, d):
    return lambda x: f(x)[d]


epsilon1 = np.array([
    [1, 0],
    [0, 0],
    [0, 1]])
epsilon2 = np.array([
    [0, 0],
    [0, 1],
    [1, 0]])
Epsilon = np.array([epsilon1, epsilon2])


def get_elasticity_A_F(p, tri, dirichlet_edges, C, f, g=None, neumann_edges=np.empty(0), Nq=4,
                       finite_element=IsogeometricLinearTriangle, tri_u=None):
    if tri_u is None:
        tri_u = tri
    ref_element_geom = finite_element.geometry.ref_element
    sf_geom = finite_element.geometry.shape_fun
    sf_geom_jac = finite_element.geometry.shape_fun_jacobian
    sf_u = finite_element.displacement.shape_fun
    sf_u_jac = finite_element.displacement.shape_fun_jacobian
    n_bar = len(np.unique(tri_u))
    degrees = 2 * n_bar
    A = np.zeros((degrees, degrees))
    F = np.zeros(degrees)

    for element, element_u in zip(tri, tri_u):
        X = p[element].T
        index_u_2d = np.concatenate((index(element_u, 0), index(element_u, 1)))

        def right_integrand(ksi):
            jacobian_det = np.linalg.det(X @ sf_geom_jac(ksi))
            return np.kron(f(X @ sf_geom(ksi)), sf_u(ksi)) * jacobian_det

        # find coefficients for basis functions
        XY = np.append(np.ones((3, 1)), p[element], axis=1)
        B = np.linalg.solve(XY, np.identity(3))

        # coordinates of the nodes of the element
        p1, p2, p3 = p[element[0]], p[element[1]], p[element[2]]
        F[index_u_2d] += qd.quadrature2D(*ref_element_geom, Nq, right_integrand)

        for da in [0, 1]:
            for db in [0, 1]:
                def left_integrand(ksi):
                    left = sf_u_jac(ksi) @ np.linalg.inv(X @ sf_geom_jac(ksi))
                    inner = (Epsilon[da] @ left.T).T @ C @ Epsilon[db] @ left.T
                    jacobian_det = np.linalg.det(X @ sf_geom_jac(ksi))
                    return inner * jacobian_det

                A[np.ix_(index(element_u, da), index(element_u, db))] += qd.quadrature2D(*ref_element_geom, 1,
                                                                                         left_integrand)

        # find a(phi_i,phi_j) and l(phi_i)
        for alpha in range(3):
            for da in [0, 1]:
                for beta in range(3):
                    for db in [0, 1]:
                        # apply neumann conditions if applicable
                        if [element[alpha], element[beta]] in neumann_edges.tolist():
                            vertex1, vertex2 = p[element[alpha]], p[element[beta]]
                            Ha = lambda x: (B[0, alpha] + B[1:3, alpha] @ x)
                            Hb = lambda x: B[0, beta] + B[1:3, beta] @ x

                            F[index(element[alpha], da)] += qd.quadrature1D(vertex1, vertex2, Nq,
                                                                            function_multiply(Ha, proj(g, da)))
                            F[index(element[beta], db)] += qd.quadrature1D(vertex1, vertex2, Nq,
                                                                           function_multiply(Hb, proj(g, db)))

    # Applying dirichlet boundary conditions
    epsilon = 1e-100
    dirichlet_vertecis = np.unique(dirichlet_edges)
    for node in dirichlet_vertecis:
        for d in [0, 1]:
            A[index(node, d), index(node, d)] = 1 / epsilon
            F[index(node, d)] = 0

    return A, F


def solve_elastic(p, tri, dirichlet_edges, C, f, g=None, neumann_edges=np.empty(0), Nq=4):
    A, F = get_elasticity_A_F(p, tri, dirichlet_edges, C, f, g, neumann_edges, Nq)
    A, F = sp.csc_matrix(A), sp.csc_matrix(F).T

    U = sp.linalg.spsolve(A, F)

    return reshape_U(U)
