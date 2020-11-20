import unittest
import numpy.testing as nptest
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from femsolver.stress_recovery import get_derivative, get_naive_stress_recovery
from femsolver.elasticity_solver import solve_elastic
from test.test_elasticity_solver import u, get_f, get_C
from test.getplate import getPlate


def Du(x):
    return np.array([
        [2 * x[0] * (x[1] ** 2 - 1), 2 * x[1] * (x[0] ** 2 - 1)],
        [2 * x[0] * (x[1] ** 2 - 1), 2 * x[1] * (x[0] ** 2 - 1)]
    ])

# analytical solution 
def sigma(x, C):
    Du_x = Du(x)
    strain_vector = np.array([
        Du_x[0, 0],
        Du_x[1, 1],
        Du_x[1, 0] + Du_x[0, 1]
    ])
    stress_vector = C @ strain_vector
    # return strain tensor
    return np.array([
        [stress_vector[0], stress_vector[2]],
        [stress_vector[2], stress_vector[1]]
    ])


class TestStressRecovery(unittest.TestCase):
    def test_get_derivative(self):
        # test some simple geometries to know that the derivative is correct

        Ps = np.array([
            [0, 0],
            [3, 0],
            [0, 2]
        ])
        U = np.array([[0, 1, 2], [0, 3, 4]]).T
        DU = np.array([
            [1 / 3, 3 / 3],
            [2 / 2, 4 / 2]
        ])
        nptest.assert_array_equal(get_derivative(U=U, p=Ps, element=[0, 1, 2]), DU)
        nptest.assert_array_equal(get_derivative(U=U, p=Ps, element=[1, 0, 2]), DU)
        nptest.assert_array_equal(get_derivative(U=U, p=Ps, element=[0, 2, 1]), DU)

        Ps2 = np.array([
            [0, 0],
            [1, 1],
            [0, 1]
        ])
        U2 = np.array([[0, 2, 1], [0, 4, 2]]).T
        DU2 = np.array([
            [1, 2],
            [1, 2]
        ])
        nptest.assert_array_equal(get_derivative(U=U2, p=Ps2, element=[0, 1, 2]), DU2)
        nptest.assert_array_equal(get_derivative(U=U2, p=Ps2, element=[1, 0, 2]), DU2)
        nptest.assert_array_equal(get_derivative(U=U2, p=Ps2, element=[0, 2, 1]), DU2)

    def test_naive_stress_calculation_analytic(self):
        # Test calculated stresss from analytical displacement u 

        N_list = 2 ** np.arange(2, 7)
        test_values = N_list ** 2 * 2
        print("\n\nComparing calculated stress to analytical stress:")
        E = 200
        v = 0.1


        for i, N in enumerate(N_list):
            p, tri, edge = getPlate(N)
            edge -= 1  # The edge indexes seem to be off
            U = np.moveaxis(u(p.T), -1, 0)
            C = get_C(E, v)

            Sigma = get_naive_stress_recovery(U, p, tri, C)
            Sigma_exact = np.moveaxis(sigma(p.T, C), -1, 0)

            max_value = np.max(abs(Sigma_exact))
            max_error = np.max(abs(Sigma_exact - Sigma))
            rel_error = max_error / max_value

            print(f"f = {test_values[i]}, rel error:", rel_error)
            self.assertAlmostEqual(rel_error, 0, delta=10 / test_values[i] ** 0.5)

    def test_naive_stress_recovery(self):
        # Test calculated stress from numerically found displacement u 
        N_list = 2 ** np.arange(2, 7)
        test_values = N_list ** 2 * 2
        print("\n\nComparing recovered stress to analytical stress:")
        E = 5
        v = 0.1
        for i, N in enumerate(N_list):
            p, tri, edge = getPlate(N)
            edge -= 1  # The edge indexes seem to be off
            C = get_C(E, v)

            U = solve_elastic(p, tri, edge, C, f=get_f(E, v))
            Sigma = get_naive_stress_recovery(U, p, tri, C)

            Sigma_exact = np.moveaxis(sigma(p.T, C), -1, 0)

            max_value = np.max(abs(Sigma_exact))
            max_error = np.max(abs(Sigma_exact - Sigma))
            rel_error = max_error / max_value
            print(f"f = {test_values[i]}, rel error:", rel_error)
            self.assertAlmostEqual(rel_error, 0, delta=10 / test_values[i] ** 0.5)

    def test_plot_naive_stress_recovery(self):
        N = 32
        E = 5
        v = 0.1
        dim = [0, 0]
        fig = plt.figure(figsize=plt.figaspect(2))
        C = get_C(E, v)

        p, tri, edge = getPlate(N)
        edge -= 1  # The edge indexes seem to be off
        U = solve_elastic(p, tri, edge, C, f=get_f(E, v))
        Sigma = get_naive_stress_recovery(U, p, tri, C)

        Sigma_exact = np.moveaxis(sigma(p.T, C), -1, 0)

        print("\n\nPlotting recovered stress")
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        if dim != [0,0]:
            ax.view_init(30, 120)
        ax.ticklabel_format(axis='x',style='sci')
        ax.set_zlabel("$\sigma_{xx,i,j}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot_trisurf(p[:, 0], p[:, 1], Sigma[:, dim[0], dim[1]], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        if dim != [0,0]:
            ax2.view_init(30, 120)
        ax2.set_zlabel("$\sigma_{xx,,i,j} - \sigma_{xx}(x_i,y_j)}$")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.plot_trisurf(p[:, 0], p[:, 1], Sigma[:, dim[0], dim[1]] - Sigma_exact[:, dim[0], dim[1]], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig(f"figures/plot_homogeneous_dirichlet_elastic_sigma_dim_{dim[0]}_{dim[1]}.pdf")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
