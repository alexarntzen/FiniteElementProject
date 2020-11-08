import unittest
import numpy.testing as nptest
import numpy as np

from femsolver.stress_recovery import get_derivative, get_naive_stress_recovery
from femsolver.elasticity_solver import solve_elastic
from test.test_elasticity_solver import u, get_f, get_C
from test.getplate import getPlate


def Du(x):
    return np.array([
        [2 * x[0] * (x[1] ** 2 - 1), 2 * x[1] * (x[0] ** 2 - 1)],
        [2 * x[0] * (x[1] ** 2 - 1), 2 * x[1] * (x[0] ** 2 - 1)]
    ])


def omega(x, C):
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

            Omega = get_naive_stress_recovery(U, p, tri, C)
            Omega_exact = np.moveaxis(omega(p.T, C), -1, 0)

            max_value = np.max(abs(Omega_exact))
            max_error = np.max(abs(Omega_exact - Omega))
            rel_error = max_error / max_value

            print(f"f = {test_values[i] }, rel error:", rel_error)
            self.assertAlmostEqual(rel_error, 0, delta=10 / test_values[i]**0.5)

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
            Omega = get_naive_stress_recovery(U, p, tri, C)

            Omega_exact = np.moveaxis(omega(p.T, C), -1, 0)

            max_value = np.max(abs(Omega_exact))
            max_error = np.max(abs(Omega_exact - Omega))
            rel_error = max_error / max_value
            print(f"f = {test_values[i] }, rel error:", rel_error)
            self.assertAlmostEqual(rel_error, 0, delta=10 / test_values[i]**0.5)


if __name__ == '__main__':
    unittest.main()
