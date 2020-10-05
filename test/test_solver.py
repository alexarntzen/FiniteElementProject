import unittest
import numpy as np
import matplotlib.pyplot as plt
from test.plot_mesh import plot_disc
from part1.solver import solve
import test.getdisc as gd

def u(x):
    return np.sin(2 * np.pi * (x[0] ** 2 + x[1] ** 2))

def f(x):
    r2 = x[0]**2 + x[1]**2
    return -8*np.pi*np.cos(2*np.pi*r2) + 16*np.pi**2*r2*np.sin(2*np.pi*r2)

def g(x):
    return 4*np.pi*np.linalg.norm(x)*np.cos(2*np.pi*np.sum(x**2))

class TestSolver(unittest.TestCase):
    def test_plot_mesh(self):
        N = 10
        p, tri, edge = gd.GetDisc(N)
        plot_disc(p,tri)

    def test_compare_analytic(self):
        test_values = 2**np.arange(2,11)
        for N in test_values:

            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gd.GetDisc(N)

            # analytical solution
            U = solve(p, tri, edge, 4, f)
            max_error = np.max(np.abs(U-u(p.T)))
            print(f"N = {N}, max error:", max_error)
            self.assertAlmostEqual(max_error,0,delta=1e2/N)

    def test_plot(self):
        N = 500
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.plot_trisurf(p[:,0],p[:,1],U)
        p, tri, edge = gd.GetDisc(N)
        U = solve(p, tri, edge, 4, f)

        ax.plot_trisurf(p[:,0],p[:,1],U)
        plt.show()


class TestSolverNeumann(unittest.TestCase):
    def test_compare_analytic(self):
        test_values = 2**np.arange(4,12)
        for N in test_values:

            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gd.GetDisc(N)
            neumann_edges = edge[(p[edge[:,0]][:, 1] > 0 ) & (p[edge[:,1]][:, 1] > 0)]
            dirichlet_edges = edge[(p[edge[:,0]][:, 1] <= 0 ) | (p[edge[:,1]][:, 1] <= 0)]


            # analytical solution
            U = solve(p, tri, dirichlet_edges, 4, f=f, g=g, neumann_edges=neumann_edges)
            max_error = np.max(np.abs(U-u(p.T)))
            print(f"N = {N}, max error:", max_error)
            self.assertAlmostEqual(max_error,0,delta=1e2/N)

    def test_plot(self):
        N = 500
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        p, tri, edge = gd.GetDisc(N)
        neumann_edges = edge[(p[edge[:, 0]][:, 1] > 0) & (p[edge[:, 1]][:, 1] > 0)]
        dirichlet_edges = edge[(p[edge[:, 0]][:, 1] <= 0) | (p[edge[:, 1]][:, 1] <= 0)]

        U = solve(p, tri, dirichlet_edges, 4, f=f, g=g, neumann_edges=neumann_edges)

        ax.plot_trisurf(p[:,0],p[:,1],U)
        plt.show()




if __name__ == '__main__':
    unittest.main()
