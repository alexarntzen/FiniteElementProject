import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import test.getplate as gp
from femsolver.elasticity_solver import solve_elastic
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

def large_to_small_mapping(N_large,N_small):
    #assuming N = 2**i + 1
    
    scale = (N_large - 1) // (N_small - 1)
    l_to_s_1D = np.arange(0,N_small)*scale
    l_to_s_2D = np.ix_(l_to_s_1D,l_to_s_1D)
    
    return l_to_s_2D


# Test the solver on the problem from task 2
class TestElasticHomogeneousDirichlet(unittest.TestCase):

    def test_compare_analytic(self):
        E = 5
        v = 0.1
        N_list = 2 ** np.arange(2, 3)
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


    def test_decreasing_h(self):

        E = 5
        v = 0.1

        i_max = 6
        N_finest = 2**i_max + 1
        N_list = 2 ** np.arange(1, i_max - 1) + 1
        test_values = N_list ** 2 * 2
        u_max = 1

        rel_errors = np.zeros(len(N_list))
        run_times = np.zeros(len(N_list))


        def get_solution(N):
            p, tri, edge = gp.getPlate(N)
            edge -= 1
            t1 = time.time()
            U = solve_elastic(p, tri, edge, C=get_C(E, v), f=get_f(E, v))
            t2 = time.time()
            return U,t2-t1

        U_finest,td = get_solution(N_finest)
        print(td)

        for i,N in enumerate(N_list):
            
            U_N,td =  get_solution(N)

            u_max = np.max(np.abs(U_N))

            r = (N_finest-1)//(N-1)
            l_to_s_index = np.array([np.arange(0+j*r*N_finest,N_finest+j*r*N_finest,r).tolist() for j in range(0,N)]).flatten()

            #max_error = np.max(np.abs(U_N[:,1] - U_finest[:,1][l_to_s_index]))
            max_error = np.linalg.norm((U_N[:,1] - U_finest[:,1][l_to_s_index]),ord=np.inf)
            rel_errors[i] = max_error
            run_times[i] = td

        element_sizes = 1/N_list

        convergence = np.polyfit(np.log(element_sizes),np.log(rel_errors),deg=1)[0]
        print(convergence)


        plt.loglog(element_sizes,rel_errors,marker="o")
        plt.ylabel("Relative error")
        plt.xlabel("Element size, h")
        plt.savefig("figures/decresing_h.pdf")
        plt.clf()

        plt.plot(element_sizes,run_times,marker="o")
        plt.ylabel("Run time (s)")
        plt.xlabel("Element size, h")
        plt.savefig("figures/decresing_h_rt.pdf")
        plt.clf()


    def test_plot(self):
        N = 16
        E = 5
        v = 0.1
        fig = plt.figure(figsize=plt.figaspect(2))

        p, tri, edge = gp.getPlate(N)
        edge -= 1  # The edge indexes seem to be off
        U = solve_elastic(p, tri, edge, C=get_C(E, v), f=get_f(E, v))

        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for Dirichlet")
        ax.set_zlabel("$U_{i,j}$")
        ax.plot_trisurf(p[:, 0], p[:, 1], U[:, 0], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.set_zlabel("$U_{x,i,j} - u_{x,(x_i,y_j)}$")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U[:, 0] - u(p.T)[0], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig("figures/plot_homogeneous_dirichlet_elastic_x.pdf")
        plt.clf()

        # plotting 2nd dimention
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for Dirichlet")
        ax.set_zlabel("$U_{i,j}$")
        ax.plot_trisurf(p[:, 0], p[:, 1], U[:, 1], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.set_zlabel("$U_{x,i,j} - u_{x,(x_i,y_j)}$")
        ax2.plot_trisurf(p[:, 0], p[:, 1], U[:, 1] - u(p.T)[1], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig("figures/plot_homogeneous_dirichlet_elastic_y.pdf")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
