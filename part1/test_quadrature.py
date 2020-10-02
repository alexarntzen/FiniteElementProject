import unittest
import numpy as np
from quadrature import quadrature1D, quadrature2D, vertices_to_area_2D
from integrate_log import get_log_integral

class TestQuadratureMethods(unittest.TestCase):
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def get_one_form(z):
        return lambda x: np.inner(z,x)

    def test_linear_1D(self):
        for a in range(-10,10):
            for b in range(a, 10):
                for Nq in range(1, 5):
                    self.assertAlmostEqual(quadrature1D(a, b, Nq, self.identity), 0.5 * (b ** 2 - a ** 2))

    def test_exponential_1D(self):
        a = 1
        b = 2
        print("Testing 1D quadrature on exponential function")
        for Nq in range(1,5):
            print(f"I_{Nq} error:", quadrature1D(a, b, Nq, np.exp) - np.exp(b) + np.exp(a))
            self.assertAlmostEqual(quadrature1D(a, b, Nq, np.exp),  np.exp(b) - np.exp(a),delta=1)

    def test_constant_2D(self):
        p1 = np.array([1, 0])
        p2 = np.array([3, 1])
        p3 = np.array([3, 2])

        g = lambda x: 3

        for Nq in [1, 3, 4]:
            self.assertAlmostEqual(quadrature2D(p1, p2, p3, Nq, g),
                                   vertices_to_area_2D(p1, p2, p3) * g(0))

    def test_linear_2D(self):
        for a in range(0, 10):
            for b in range(0, 10):
                p1 = np.array([1, 3])
                p2 = p1 + np.array([0, 1])
                p3 = p1 + np.array([1, 0])
                g = self.get_one_form([a,b])
                min_val = np.min([g(p1), g(p2), g(p3)])
                diff_val = np.max([g(p1), g(p2), g(p3)]) - min_val
                for Nq in [1,3,4]:
                    self.assertAlmostEqual(quadrature2D(p1, p2, p3, Nq, g), ((a+b)/3 + min_val)*vertices_to_area_2D(p1, p2, p3))
            print("")

    def test_logarithm_2D(self):
        p1 = np.array([1, 0])
        p2 = np.array([3, 1])
        p3 = np.array([3, 2])
        g  = lambda x: np.log(np.sum(x))
        I_analytic = get_log_integral()
        print("Testing 1D quadrature on log function")
        for Nq in [1, 3, 4]:
            print(f"I_{Nq} error:", quadrature2D(p1, p2, p3, Nq, g) - I_analytic)
            self.assertAlmostEqual(quadrature2D(p1, p2, p3, Nq, g),I_analytic, delta=1/Nq)


if __name__ == '__main__':
    unittest.main()
