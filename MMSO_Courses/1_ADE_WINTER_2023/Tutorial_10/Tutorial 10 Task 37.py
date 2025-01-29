import numpy as np
import matplotlib.pyplot as plt 

def Euler(f, t, u0):
    '''
    1 stage Euler method for ODE systems.
    f   : right hand side of the ODE system.
    t   : time vector.
    u0  : vector of initial values.
    return: matrix U where U[i, :] are the solutions for u^(i)
    '''
    n = len(t)
    m = len(u0)
    U = np.zeros((m, n))
    U[:, 0] = u0
    
    for j in range(n-1):
        h = t[j+1] - t[j]
        k1 = np.array(f(t[j], U[:, j]))
        U[:, j+1] = U[:, j] + h * k1
    return U

def RungeKutta(f, t, u0):
    '''
    4 stage Runge Kutta method for ODE systems.
    f   : right hand side of the ODE system.
    t   : time vector.
    u0  : vector of initial values.
    return: matrix U where U[i, :] are the solutions for u^(i)
    '''
    n = len(t)
    m = len(u0)
    U = np.zeros((m, n))
    U[:, 0] = u0
    for j in range(n-1):
        h = t[j+1] - t[j]
        k1 = np.array(f(t[j], U[:, j]))
        k2 = np.array(f(t[j] + (h/2), U[:, j] + k1 * (h/2)))
        k3 = np.array(f(t[j] + (h/2), U[:, j] + k2 * (h/2)))
        k4 = np.array(f(t[j] + h, U[:, j] + k3 * h))
        U[:, j+1] = U[:, j] + (k1 + 2*k2 + 2*k3 + k4) * (h/6)
    return U

def mathpendulum(t, u):
    '''
    Mathematical pendulum ODE system u'' + sin(u) = 0
    solved using reduction of order.
    '''
    y0 = u[0]
    y1 = u[1]
    y1prime = -np.sin(y0)
    return [y1, y1prime]

def linearizedpendulum(t, u):
    '''
    Linearized pendulum for small values of u
    '''
    y0 = u[0]
    y1 = u[1]
    y1prime = -y0
    return [y1, y1prime]

def H(u,uprime):
    '''
    Hamiltonian of the pendulum ODE.
    Can be evaluated after u and uprime have been found.
    '''
    return (uprime)**2 / 2 - np.cos(u)

def V(u,uprime):
    '''
    Energy of the linearized pendulum.
    '''
    return (1/2) * (u**2 + uprime**2)

def mathematicalpendulumevaluation():
    t0 = 0     # initial time
    u0 = [1,0] # initial values for u' and u''
    
    t = np.linspace(t0,25,10000)
    eulersol = Euler(mathpendulum,t,u0)   # storing solutions for u and u'
    rksol = RungeKutta(mathpendulum,t,u0) # storing solutions for u and u'
    Hamilt_e = H(eulersol[0],eulersol[1])
    Hamilt_r = H(rksol[0], rksol[1])
    return t, eulersol, rksol, Hamilt_e, Hamilt_r

def linearizedpendulumevaluation():
    t0 = 0
    u0 = [1, 0]
    t = np.linspace(t0,25,10000)
    eulersol= Euler(linearizedpendulum,t,u0)
    rksol= RungeKutta(linearizedpendulum,t,u0)
    V_e= V(eulersol[1],eulersol[1])
    V_r= V(rksol[0],rksol[1])
    return t,eulersol,rksol,V_e,V_r

def plottingHamiltonian():
    t,e,r,he,hr = mathematicalpendulumevaluation()
    plt.subplot(2, 3, 1)
    plt.plot(t, e[0])
    plt.title('u Euler')
    plt.subplot(2, 3, 2)
    plt.plot(t, e[1])
    plt.title('uprime Euler')
    plt.subplot(2, 3, 3)
    plt.plot(t, he)
    plt.title('Hamiltonian Euler')
    plt.subplot(2, 3, 4)
    plt.plot(t, r[0])
    plt.title('u Runge')
    plt.subplot(2, 3, 5)
    plt.plot(t,r[1])
    plt.title('uprime Runge')
    plt.subplot(2 ,3 ,6)
    plt.plot(t ,hr )
    plt.title ('Hamiltonian Runge')
    plt.tight_layout ()
    plt.show ()
    
def plottingEnergy():
    t,e,r,ve,vr = linearizedpendulumevaluation()
    plt.subplot(2, 3, 1)
    plt.plot(t, e[0])
    plt.title('u Euler')
    plt.subplot(2, 3, 2)
    plt.plot(t, e[1])
    plt.title('uprime Euler')
    plt.subplot(2, 3, 3)
    plt.plot(t, ve)
    plt.title('Energy Euler')
    plt.subplot(2, 3, 4)
    plt.plot(t, r[0])
    plt.title('u Runge')
    plt.subplot(2, 3, 5)
    plt.plot(t, r[1])
    plt.title('uprime Runge')
    plt.subplot(2, 3 ,6)
    plt.plot(t,vr)
    plt.title('Energy Runge')
    plt.tight_layout()
    plt.show()

plottingHamiltonian()
plottingEnergy()



