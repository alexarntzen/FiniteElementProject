import numpy as np

import femsolver.quadrature as qd


def compose(f, g):
    return lambda x: f(g(x))


def function_multiply(f, g):
    return lambda x: f(x) * g(x)


def get_A_F(p, tri, dirichlet_edges, f, g=None, neumann_edges=np.empty(0),Nq=4):
    n_bar = len(p)
    A = np.zeros((n_bar, n_bar))
    F = np.zeros(n_bar)

    for element in tri:

        # find coefficients for basis functions
        XY = np.append(np.ones((3, 1)), p[element], axis=1)
        C = np.linalg.solve(XY, np.identity(3))

        # coordinates of the nodes of the element
        p1, p2, p3 = p[element[0]], p[element[1]], p[element[2]]

        # find a(phi_i,phi_j) and l(phi_i)
        for alpha in range(3):

            # finding F vector
            Ha = lambda x: (C[0, alpha] + C[1:3, alpha] @ x)
            F_a = qd.quadrature2D(p1, p2, p3, Nq, function_multiply(Ha, f))
            F[element[alpha]] += F_a

            for beta in range(3):

                # finding A matrix
                HaHb_derivative = lambda x: C[1:3, alpha].T @ C[1:3, beta]
                I_ab = qd.quadrature2D(p1, p2, p3, 1, HaHb_derivative)
                A[element[alpha], element[beta]] += I_ab

                # apply neumann conditions if applicable
                if [element[alpha], element[beta]] in neumann_edges.tolist():
                    vertex1, vertex2 = p[element[alpha]], p[element[beta]]
                    Hb = lambda x: (C[0, beta] + C[1:3, beta] @ x)

                    F[element[alpha]] += qd.quadrature1D(vertex1, vertex2, Nq, function_multiply(Ha, g))
                    F[element[beta]] += qd.quadrature1D(vertex1, vertex2, Nq, function_multiply(Hb, g))

    # Applying dirichlet boundary conditions
    epsilon = 1e-100
    dirichlet_vertecis = np.unique(dirichlet_edges)
    for i in dirichlet_vertecis:
        A[i, i] = 1 / epsilon
        F[i] = 0

    return A, F


def solve(p, tri, dirichlet_edges, f, g=None, neumann_edges=np.empty(0),Nq=4):
    A, F = get_A_F(p, tri, dirichlet_edges, f, g, neumann_edges,Nq,)
    U = np.linalg.solve(A, F)
    return U
