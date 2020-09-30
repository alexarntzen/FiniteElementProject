import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import getdisc as gd

N = 200
p,tri,edge = gd.GetDisc(N)

def plot_disc(p,tri):
    line_segments = np.zeros((len(tri),4,2))
    for i,t in enumerate(tri):
        for j in range(3):
            line_segments[i,j,:] = p[t[j]]
        line_segments[i,3,:] = p[t[0]]
    
    plt.scatter(p[:,0],p[:,1])
    plt.gca().add_collection(LineCollection(line_segments))
    plt.show()
    
plot_disc(p,tri)