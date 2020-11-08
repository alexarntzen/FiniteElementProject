import unittest
import numpy.testing as nptest
import numpy as np

from femsolver.stress_recovery import get_derivative


class TestStressRecovery(unittest.TestCase):
    # test the most simplest
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

    test_get_derivative[]


if __name__ == '__main__':
    unittest.main()
