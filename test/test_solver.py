import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import test.getdisc as gd
from part1.solver import solve, get_A_F


def u(x):
    return np.sin(2 * np.pi * (x[0] ** 2 + x[1] ** 2))


def f(x):
    r2 = x[0] ** 2 + x[1] ** 2
    return -8 * np.pi * np.cos(2 * np.pi * r2) + 16 * np.pi ** 2 * r2 * np.sin(2 * np.pi * r2)


def g(x):
    return 4 * np.pi * np.linalg.norm(x) * np.cos(2 * np.pi * np.sum(x ** 2))


class TestSingularity(unittest.TestCase):
    def test_singularity(self):
        test_values = 2 ** np.arange(4, 11)
        print("\n\n\Checking matrix is singular before applying boundary conditions")
        for N in test_values:
            p, tri, edge = gd.GetDisc(N)
            A, F = get_A_F(p, tri, [], 4, f)
            n = np.shape(A)[0]
            rank = np.linalg.matrix_rank(A)
            print("Rank of A: ", rank, ". Size of matrix A :(", n, " x ", n, ")")
            self.assertNotEqual(n, rank)


class TestHomogeneousDirichlet(unittest.TestCase):

    def test_compare_analytic(self):
        test_values = 2 ** np.arange(4, 11)
        rel_errors = np.zeros(len(test_values))
        u_max = 1
        print("\n\nComparing homogeneous dirichlet to analytical result")
        for i, N in enumerate(test_values):
            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gd.GetDisc(N)
            u_max = np.max(np.abs(u(p.T)))

            # numerical solution
            U = solve(p, tri, edge, 4, f)

            max_error = np.max(np.abs(U - u(p.T)))
            print(f"N = {N}, max error:", max_error)
            self.assertAlmostEqual(max_error, 0, delta=1e2 / N)
            rel_errors[i] = max_error

        rel_errors /= u_max
        fig = plt.figure(figsize=plt.figaspect(1))
        plt.loglog(test_values, rel_errors, marker="o")

        # plt.title("Convergence of relative error for Dirichlet")
        plt.ylabel("Relative error")
        plt.xlabel("$N$ nodes in mesh")
        plt.savefig("figures/convergence_homogeneous_dirichlet.pdf")
        plt.clf()

    def test_plot(self):
        N = 1024

        fig = plt.figure(figsize=plt.figaspect(2))

        p, tri, edge = gd.GetDisc(N)
        U = solve(p, tri, edge, 4, f)
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for Dirichlet")
        ax.set_zlabel("$U_{i,j}$")
        ax.plot_trisurf(p[:, 0], p[:, 1], U, cmap=cm.viridis)

        # ax = fig.add_subplot(3, 1, 2, projection='3d')
        # ax.set_title("Analytical solution")
        # ax.set_zlabel("$U_{i,j}$")
        # ax.plot_trisurf(p[:,0],p[:,1],u(p.T),cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.set_zlabel("$U_{i,j} - u(x_i,y_j)$")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U - u(p.T), cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig("figures/plot_homogeneous_dirichlet.pdf")
        plt.clf()


class TestSolverNeumann(unittest.TestCase):
    def test_compare_analytic(self):
        test_values = 2 ** np.arange(4, 12)
        rel_errors = np.zeros(len(test_values))
        u_max = 1
        print("\n\nComparing mixed Neumann to analytical result")
        for i, N in enumerate(test_values):
            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gd.GetDisc(N)
            neumann_edges = edge[(p[edge[:, 0]][:, 1] > 0) & (p[edge[:, 1]][:, 1] > 0)]
            dirichlet_edges = edge[(p[edge[:, 0]][:, 1] <= 0) | (p[edge[:, 1]][:, 1] <= 0)]
            u_max = np.max(np.abs(u(p.T)))

            # numerical solution
            U = solve(p, tri, dirichlet_edges, 4, f=f, g=g, neumann_edges=neumann_edges)
            max_error = np.max(np.abs(U - u(p.T)))
            print(f"N = {N}, max error:", max_error)
            self.assertAlmostEqual(max_error, 0, delta=1e2 / N)
            rel_errors[i] = max_error

        rel_errors /= u_max

        fig = plt.figure(figsize=plt.figaspect(1))
        plt.loglog(test_values, rel_errors, marker="o")
        # plt.title("Convergence of relative error for partial Neumann")
        plt.ylabel("Relative error")
        plt.xlabel("$N$ nodes in mesh")
        plt.savefig("figures/convergence_mixed_neumann.pdf")
        plt.clf()

    def test_plot(self):
        N = 1024

        fig = plt.figure(figsize=plt.figaspect(2))

        p, tri, edge = gd.GetDisc(N)
        neumann_edges = edge[(p[edge[:, 0]][:, 1] > 0) & (p[edge[:, 1]][:, 1] > 0)]
        dirichlet_edges = edge[(p[edge[:, 0]][:, 1] <= 0) | (p[edge[:, 1]][:, 1] <= 0)]

        U = solve(p, tri, dirichlet_edges, 4, f, g, neumann_edges)

        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for mixed Neumann")
        ax.set_zlabel("$U_{i,j}$")
        ax.plot_trisurf(p[:, 0], p[:, 1], U, cmap=cm.viridis)

        # ax = fig.add_subplot(3, 1, 2, projection='3d')
        # ax.set_title("Analytical solution")
        # ax.set_zlabel("$U_{i,j}$")
        # ax.plot_trisurf(p[:,0],p[:,1],u(p.T),cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.set_zlabel("$U_{i,j} - u(x_i,y_j)$")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U - u(p.T), cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig("figures/plot_mixed_neumann.pdf")
        fig.clf()


if __name__ == '__main__':
    unittest.main()
