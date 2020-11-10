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
    # test some simple geometries
    def test_get_derivative(self):
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
        N_list = 2 ** np.arange(2, 7)
        test_values = N_list ** 2 * 2
        print("\n\nComparing calculated stress to analytical stress:")
        E = 5
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
        N = 16
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
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        # ax.set_title("Numerical solution for Dirichlet")
        ax.set_zlabel("$\Sigma_{i,j}$")
        ax.plot_trisurf(p[:, 0], p[:, 1], Sigma[:, dim[0], dim[1]], cmap=cm.viridis)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title("Error")
        ax2.set_zlabel("$\Sigma_{x,i,j} - \sigma_{x,(x_i,y_j)}$")
        ax2.plot_trisurf(p[:, 0], p[:, 1], Sigma_exact[:, dim[0], dim[1]], cmap=cm.viridis)
        # ax.plot_trisurf(p[:, 0], p[:, 1], U)
        plt.savefig(f"figures/plot_homogeneous_dirichlet_elestic_sigma_dim_{dim[0]}_{dim[1]}.pdf")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
