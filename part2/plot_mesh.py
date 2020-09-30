import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import getdisc as gd

N = 100
p,tri,edge = gd.GetDisc(N)

def plot_disc(p,tri):
    new_tri = np.concatenate((
        tri,np.array([tri[:,0].T]).T),axis=1)
    plt.scatter(p[:,0],p[:,1])
    plt.gca().add_collection(LineCollection(p[new_tri]))
    plt.show()
    
#plot_disc(p,tri)

