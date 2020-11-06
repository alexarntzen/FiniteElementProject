import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_disc(p,tri,plt=plt):
    new_tri = np.concatenate((
        tri,np.array([tri[:,0].T]).T),axis=1)
    plt.scatter(p[:,0],p[:,1])
    plt.gca().add_collection(LineCollection(p[new_tri]))

def plot_disc_with_edegs(p,tri,edges,plt=plt):
    new_tri = np.concatenate((
        tri,np.array([tri[:,0].T]).T),axis=1)
    plt.scatter(p[:,0],p[:,1])
    plt.gca().add_collection(LineCollection(p[new_tri]))
    plt.gca().add_collection(LineCollection(p[edges],colors="r"))

