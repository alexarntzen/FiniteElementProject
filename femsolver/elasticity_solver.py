import numpy as np

import femsolver.quadrature as qd


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


Epsilon = (
    np.array([
        [1, 0],
        [0, 0],
        [0, 1]
    ]),
    np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ])
)


def get_elasticity_A_F(p, tri, dirichlet_edges, C, f, g=None, neumann_edges=np.empty(0), Nq=4):
    n_bar = len(p)
    degrees = 2 * n_bar
    A = np.zeros((degrees, degrees))
    F = np.zeros(degrees)

    for element in tri:

        # find coefficients for basis functions
        XY = np.append(np.ones((3, 1)), p[element], axis=1)
        B = np.linalg.solve(XY, np.identity(3))

        # coordinates of the nodes of the element
        p1, p2, p3 = p[element[0]], p[element[1]], p[element[2]]

        # find a(phi_i,phi_j) and l(phi_i)
        for alpha in range(3):
            for da in [0, 1]:

                # finding F vector
                Ha = lambda x: (B[0, alpha] + B[1:3, alpha] @ x)
                F_a = qd.quadrature2D(p1, p2, p3, Nq, function_multiply(Ha, proj(f, da)))
                F[index(element[alpha], da)] += F_a
                index(element[alpha], da)
                for beta in range(3):
                    for db in [0, 1]:
                        # finding A matrix
                        HaHb_derivative = lambda x: (Epsilon[da] @ B[1:3, alpha]).T @ C @ (Epsilon[db] @ B[1:3, beta])
                        I_ab = (Epsilon[da] @ B[1:3, alpha]).T @ C @ (Epsilon[db] @ B[1:3, beta])*qd.vertices_to_area_2D(p1,p2,p3)
                        A[index(element[alpha], da), index(element[beta], db)] += I_ab
                        # apply neumann conditions if applicable

                        if [element[alpha], element[beta]] in neumann_edges.tolist():
                            vertex1, vertex2 = p[element[alpha]], p[element[beta]]
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
    U = np.linalg.solve(A, F)
    return reshape_U(U)
