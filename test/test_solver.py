import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from part1.solver import solve
import test.getdisc as gd

def u(x):
    return np.sin(2 * np.pi * (x[0] ** 2 + x[1] ** 2))

def f(x):
    r2 = x[0]**2 + x[1]**2
    return -8*np.pi*np.cos(2*np.pi*r2) + 16*np.pi**2*r2*np.sin(2*np.pi*r2)

class TestSolver(unittest.TestCase):
    def test_compare_analytic(self):
        test_values = 2**np.arange(4,13)
        rel_errors = np.zeros(len(test_values))
        

        for i,N in enumerate(test_values):

            # p is coordinates of all nodes
            # tri is a list of indicies (rows in p) of all nodes belonging to one element
            # edge is lists of all nodes on the edge
            p, tri, edge = gd.GetDisc(N)
            u_max = np.max(np.abs(u(p.T)))

            # analytical solution
            U = solve(p, tri, edge, 4, f)

            max_error = np.max(np.abs(U-u(p.T)))
            print(f"N = {N}, max error:", max_error)
            self.assertAlmostEqual(max_error,0,delta=1e2/N)
            rel_errors[i] = max_error

        rel_errors /= u_max

        order = np.polyfit(np.log(test_values),np.log(rel_errors),1)[0]
        print("Order of convergence = ", order)
        plt.loglog(test_values,rel_errors,marker="o")
        
        plt.title("Convergence of relative error")
        plt.ylabel("Relative error")
        plt.xlabel("$N$ nodes in mesh")  
        plt.show()
        

        

    def test_plot(self):
        N = 1024

        fig = plt.figure(figsize=plt.figaspect(2))

        p, tri, edge = gd.GetDisc(N)
        U = solve(p, tri, edge, 4, f)

        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title("Numerical solution")
        ax.set_zlabel("$U_{i,j}$")
        ax.plot_trisurf(p[:,0],p[:,1],U,cmap=cm.viridis)

        # ax = fig.add_subplot(3, 1, 2, projection='3d')
        # ax.set_title("Analytical solution")
        # ax.set_zlabel("$U_{i,j}$")
        # ax.plot_trisurf(p[:,0],p[:,1],u(p.T),cmap=cm.viridis)

        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.set_title("Error")
        ax.set_zlabel("$U_{i,j} - u(x_i,y_j)$")
        ax.plot_trisurf(p[:,0],p[:,1],U-u(p.T),cmap=cm.viridis)
        plt.show()






if __name__ == '__main__':
    unittest.main()
