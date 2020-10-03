import numpy as np
import part1.quadrature as qd

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


