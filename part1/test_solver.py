import unittest
from solver import solve
import numpy as np
import getdisc as gd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def u(x):
    return np.sin(2 * np.pi * (x[0] ** 2 + x[1] ** 2))

def f(x):
    r2 = x[0]**2 + x[1]**2
    return -8*np.pi*np.cos(2*np.pi*r2) + 16*np.pi**2*r2*np.sin(2*np.pi*r2)

class TestSolver(unittest.TestCase):
    def test_compare_analytic(self):
        for N in range(500, 1000, 50):

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
        N = 300
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.plot_trisurf(p[:,0],p[:,1],U)
        p, tri, edge = gd.GetDisc(N)
        U = solve(p, tri, edge, 4, f)

        ax.plot_trisurf(p[:,0],p[:,1],U-u(p.T))
        plt.show()






if __name__ == '__main__':
    unittest.main()
