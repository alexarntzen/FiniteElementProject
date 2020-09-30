import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import getdisc as gd
import plot_mesh as pm
import quadrature as qd

N = 300

#p is coordinates of all nodes
#tri is a list of indicies (rows in p) of all nodes belonging to one element
#edge is lists of all nodes on the edge
p,tri,edge = gd.GetDisc(N)
#pm.plot_disc(p, tri)


#right hand side of the equation  "load"
def f(x):
    r2 = x[0]**2 + x[1]**2
    return -8*np.pi*np.cos(2*np.pi*r2) + 16*np.pi**2*r2*np.sin(2*np.pi*r2)    

def solve(p,tri,edge,Nq,f):
    n_bar = len(p)
    A = np.zeros((n_bar,n_bar))
    F = np.zeros(n_bar)
    
    
    for element in tri:
        
        #find coefficients for basis functions
        XY = np.ones((3,3))
        C = np.zeros((3,3))
        XY[:,1:] = p[element]
        
        for i in range(3):
            b = np.zeros(3)
            b[i] = 1
            C[:,i] = np.linalg.solve(XY,b)
        
        #coordinates of the nodes of the element
        p1,p2,p3 = p[element[0]],p[element[1]],p[element[2]]
        
        #find a(phi_i,phi_j) and l(phi_i)
        for alpha in range(3):
            
            #finding F vector
            fHa = lambda x: (C[0,alpha] + C[1,alpha]*x[0] + C[2,alpha]*x[1])*f(x)
            F_a = qd.quadrature2D(p1, p2, p3, Nq, fHa)
            F[element[alpha]] += F_a
            
            for beta in range(3): 
                
                #finding A matrix
                HaHb_derivative = lambda x: C[1,alpha]*C[1,beta] + C[2,alpha]*C[2,beta]
                I_ab = qd.quadrature2D(p1, p2, p3, Nq, HaHb_derivative)
                A[element[alpha],element[beta]] += I_ab
                
    #Applying dirichlet boundary conditions
    epsilon = 1e-15
    for i in edge[:,0]:
        A[i,i] = 1/epsilon
        F[i] = 0
    
    #solving AU = F
    U = np.linalg.solve(A,F)
        
    return U

U = solve(p,tri,edge,4,f)

#analytical solution
def u(x):
    return np.sin(2*np.pi*(x[0]**2+x[1]**2))

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_trisurf(p[:,0],p[:,1],U)

ax.plot_trisurf(p[:,0],p[:,1],U-u(p.T))

