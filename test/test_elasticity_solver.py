import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import test.getplate as gp
from femsolver.elasticity_solver import solve_elastic, get_elasticity_A_F, reshape_U
import time


def u(x):
    return np.array([
        (x[0] ** 2 - 1) * (x[1] ** 2 - 1),
        (x[0] ** 2 - 1) * (x[1] ** 2 - 1)
    ])


def get_f(E, v):
    f = lambda x: E / (1 - v ** 2) * np.array([
        -2 * x[1] ** 2 - x[0] ** 2 + v * x[0] ** 2 - 2 * v * x[0] * x[1] - 2 * x[0] * x[1] + 3 - v,
        -2 * x[0] ** 2 - x[1] ** 2 + v * x[1] ** 2 - 2 * v * x[0] * x[1] - 2 * x[0] * x[1] + 3 - v
    ])
    return f


def get_C(E, v):
    return E / (1 - v ** 2) * np.array([
        [1, v, 0],
        [v, 1, 0],
        [0, 0, (1 - v) / 2]
    ])


# Test the solver on the problem from task 2
class TestElasticHomogeneousDirichlet(unittest.TestCase):

    def test_compare_analytic(self):
        E = 5
        v = 0.1
        N_list = 2 ** np.arange(2, 7)
        test_values = N_list ** 2 * 2
        rel_errors = np.zeros(len(N_list))
        u_max = 1
        print("\n\nComparing homogeneous dirichlet elastic solution to analytical result")
        for i, N in enumerate(N_list):
            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gp.getPlate(N)
            u_max = np.max(np.abs(u(p.T).T))
            edge -= 1
            U = solve_elastic(p, tri, edge, C=get_C(E, v), f=get_f(E, v))

            max_error = np.max(np.abs(U - u(p.T).T))
            print(f"f = {test_values[i]}, max error:", max_error)
            self.assertAlmostEqual(max_error, 0, delta=10 / test_values[i])
            rel_errors[i] = max_error

        rel_errors /= u_max
        fig = plt.figure(figsize=plt.figaspect(1))
        plt.loglog(test_values, rel_errors, marker="o")

        # plt.title("Convergence of relative error for Dirichlet")
        plt.ylabel("Relative error")
        plt.xlabel("Degrees of freedom")
        plt.savefig("figures/convergence_homogeneous_dirichlet_elastic.pdf")
        plt.clf()

    def test_plot(self):
        N = 32
        E = 200
        v = 0.1

        p, tri, edge = gp.getPlate(N)
        edge -= 1  # The edge indexes seem to be off
        U = solve_elastic(p, tri, edge, C=get_C(E, v), f=get_f(E, v))

        print("\n\nGenrating plot for homogeneous linear elasticity problem")
        fig = plt.figure(figsize=plt.figaspect(2))
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_zlabel("$U_{i,j}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot_trisurf(p[:, 0], p[:, 1], U[:, 0], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.zaxis._axinfo['label']['space_factor'] = 4
        # ax2.set_title("Error")
        ax2.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

        ax2.set_zlabel("$e_{x,i,j}$")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U[:, 0] - u(p.T)[0], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)

        plt.savefig("figures/plot_homogeneous_dirichlet_elastic_x.pdf")

        plt.clf()

        # plotting 2nd dimention
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for Dirichlet")
        ax.set_zlabel("$U_{i,j}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot_trisurf(p[:, 0], p[:, 1], U[:, 1], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
        ax2.set_zlabel("$e_{y,i,j}$")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U[:, 1] - u(p.T)[1], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)

        plt.savefig("figures/plot_homogeneous_dirichlet_elastic_y.pdf")
        plt.clf()


def large_to_small_mapping(N_large, N_small):
    # assuming N = 2**i + 1

    scale = (N_large - 1) // (N_small - 1)
    l_to_s_1D = np.arange(0, N_small) * scale
    l_to_s_2D = np.ix_(l_to_s_1D, l_to_s_1D)

    return l_to_s_2D


# Functions for timing of solution time
def solve_LU(A, F):
    U = np.linalg.solve(A, F)
    return reshape_U(U)


def solve_sparse(A, F):
    A_sp = sp.csr_matrix(A)
    U = splin.spsolve(A_sp, F)
    return reshape_U(U)


def solve_cg(A, F):
    A_sp = sp.csr_matrix(A)
    U = splin.cg(A_sp, F, atol=0.1)[0]
    return reshape_U(U)


# get the solution and computaiton time
def get_solution(N, E, v):
    # p is coordinates of all nodes
    # tri is a list of indicies (rows in p) of all nodes belonging to one element
    # edge is lists of all nodes on the edge
    p, tri, edge = gp.getPlate(N)
    edge -= 1

    # Using time.perf_counter() to find the runtime of solve_elastic()
    t1 = time.perf_counter()
    A, F = get_elasticity_A_F(p, tri, edge, C=get_C(E, v), f=get_f(E, v))
    t2 = time.perf_counter()
    make_time = t2 - t1

    t1 = time.perf_counter()
    U = solve_LU(A, F)
    t2 = time.perf_counter()
    lu_time = t2 - t1

    t1 = time.perf_counter()
    U = solve_sparse(A, F)
    t2 = time.perf_counter()
    sparse_time = t2 - t1

    t1 = time.perf_counter()
    U = solve_cg(A, F)
    t2 = time.perf_counter()
    cg_time = t2 - t1

    print(lu_time)
    print(sparse_time)

    return U, make_time, lu_time, sparse_time, cg_time


class TestElasticSolverPerformance(unittest.TestCase):

    # Test the solver for a problem with small element sizes
    # and compare with solutions on a coarser mesh
    def test_decreasing_h(self):
        E = 5
        v = 0.1

        # Maximum N / minimum stepsize, h = 1/(2**i_max)
        i_max = 6
        N_finest = 2 ** i_max + 1
        N_list = 2 ** np.arange(1, i_max - 1) + 1

        test_values = N_list ** 2 * 2
        u_max = 1

        rel_errors = np.zeros(len(N_list))
        make_times = np.zeros(len(N_list) + 1)
        lu_times = np.zeros(len(N_list) + 1)
        sparse_times = np.zeros(len(N_list) + 1)
        cg_times = np.zeros(len(N_list) + 1)

        print(f"\n\nComparing elastic solution for h_small = {1 / N_finest:.3f} to larger step sizes.")

        U_finest, make_time, lu_time, sparse_time, cg_time = get_solution(N_finest, E, v)
        make_times[-1] = make_time
        lu_times[-1] = lu_time
        sparse_times[-1] = sparse_time
        cg_times[-1] = cg_time

        for i, N in enumerate(N_list):
            U_N, make_time, lu_time, sparse_time, cg_time = get_solution(N, E, v)
            u_max = np.max(np.abs(U_N))

            # getting points that have same location in fine and coarse mesh
            r = (N_finest - 1) // (N - 1)
            l_to_s_index = np.array([np.arange(0 + j * r * N_finest, N_finest + j * r * N_finest, r).tolist() for j in
                                     range(0, N)]).flatten()
            max_error = np.max(np.abs(U_N[:, 0] - U_finest[:, 0][l_to_s_index]))
            print(f"h = {1 / N:.2f}, max error: {max_error:.5f} run time: {sum([make_time + sparse_time]):.3f}")
            self.assertAlmostEqual(max_error, 0, delta=10 / test_values[i])

            rel_errors[i] = max_error
            make_times[i] = make_time
            lu_times[i] = lu_time
            sparse_times[i] = sparse_time
            cg_times[i] = cg_time

        N_list = np.append(N_list, N_finest)
        element_sizes = 1 / N_list

        error_convergence = np.polyfit(np.log(element_sizes[:-1]), np.log(rel_errors), deg=1)[0]

        print("\n\nGenerating plot for solution convergence and time complexity for the linear elasticity problem ")

        plt.gcf().subplots_adjust(left=0.15)
        # plt.title("Relative deviance for different element sizes")
        plt.loglog(element_sizes[:-1], rel_errors, marker="o")

        plt.ylabel(" $|| U_f - U_h ||_{\infty}$")
        plt.xlabel("Element size, $h$")
        plt.savefig("figures/decreasing_h_error.pdf")
        plt.clf()


        print(lu_times)
        print(sparse_times)

        time_convergence_make = np.polyfit(np.log(element_sizes), np.log(make_times), deg=1)[0]
        time_convergence_LU = np.polyfit(np.log(element_sizes), np.log(lu_times), deg=1)[0]
        time_convergence_sparse = np.polyfit(np.log(element_sizes), np.log(sparse_times), deg=1)[0]
        time_convergence_cg = np.polyfit(np.log(element_sizes), np.log(cg_times), deg=1)[0]
        print(f"\nEstimated time complexity for make system: O(h^{time_convergence_make}) ")
        print(f"Estimated time complexity of solve LU: O(h^{time_convergence_LU}) ")
        print(f"Estimated time complexity of solve sparse: O(h^{time_convergence_sparse}) ")
        print(f"Estimated time complexity of of solve conjugate gradient: O(h^{time_convergence_cg}) ")

        plt.gcf().subplots_adjust(left=0.15)
        # plt.title("Runtime for different element sizes")
        plt.loglog(element_sizes, make_times, marker="o", label="make system")
        plt.loglog(element_sizes, lu_times, marker="o", label="solve LU")
        plt.loglog(element_sizes, sparse_times, marker="o", label="solve sparse")
        plt.loglog(element_sizes, cg_times, marker="o", label="solve conjugate gradient")
        plt.legend()
        plt.ylabel("Run time ($s$)")
        plt.xlabel("Element size, $h$")
        plt.savefig("figures/decreasing_h_runtime.pdf")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
