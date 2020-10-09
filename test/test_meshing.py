import unittest

import matplotlib.pyplot as plt
import numpy as np

import test.getdisc as gd
from test.plot_mesh import plot_disc


class TestMeshing(unittest.TestCase):
    def test_plot_disc_mesh(self):
        print("\n\nPlotting meshes")
        n_list = 2 ** np.array([4, 6, 8])
        fig, axis = plt.subplots(1, len(n_list), figsize=plt.figaspect(1/3))
        for i in range(len(n_list)):
            p, tri, edge = gd.GetDisc(n_list[i])
            plt.sca(axis[i])
            plot_disc(p, tri, plt)
        plt.savefig("figures/plot_meshes.pdf")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
