# Task 38 Absolute Value ODE

import numpy as np
import matplotlib.pyplot as plt

def Euler(f,t,u0):
    '''
    1 stage Euler method for ODE systems.
    f     : right hand side of the ODE system.
    t     : time vector.
    u0    : vector of initial values.
    return: matrix U where U[i,:] are the solutions for u^(i)
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

def RungeKutta2(f,t,u0):
    '''
    2 stage Runge Kutta method for ODE systems.
    f     : right hand side of the ODE system.
    t     : time vector.
    u0    : vector of initial values.
    return: matrix U where U[i,:] are the solutions for u^(i)
    '''
    n = len(t)
    m = len(u0)
    U = np.zeros((m, n))
    U[:, 0] = u0
    for j in range(n-1):
        h = t[j+1] - t[j]
        k1 = np.array(f(t[j], U[:, j]))
        k2 = np.array(f(t[j] + h, U[:, j] + k1 * h))
        U[:, j+1] = U[:, j] + (k1 + k2) * (h/2)
    return U

def RungeKutta4(f,t,u0):
    '''
    4 stage Runge Kutta method for ODE systems.
    f     : right hand side of the ODE system.
    t     : time vector.
    u0    : vector of initial values.
    return: matrix U where U[i,:] are the solutions for u^(i)
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

def f(t,u):
    '''
    Right hand side of the ODE.
    '''
    return np.abs(1.1-u) + 1

def u(t):
    '''
    The analytical solution
    '''
    return 2.1 - 1.1 * np.exp(-t)

def evaluation():
    t0 = 0   # initial time
    u0 = [1] # initial values for u' and u''
    T  = 0.1
    t1        = np.linspace(t0,T,int(1/0.1))   # step size 0.1
    t2        = np.linspace(t0,T,int(1/0.01))  # step size 0.01
    t3        = np.linspace(t0,T,int(1/0.001))
    ex1       = u(t1)
    ex2       = u(t2)
    ex3       = u(t3)
    eulersol1 = Euler(f,t1,u0)[0,:]
    eulersol2 = Euler(f,t2,u0)[0,:]
    eulersol3 = Euler(f,t3,u0)[0,:]
    rk2sol1   = RungeKutta2(f,t1,u0)[0,:]
    rk2sol2   = RungeKutta2(f,t2,u0)[0,:]
    rk2sol3   = RungeKutta2(f,t3,u0)[0,:]
    rk4sol1   = RungeKutta4(f,t1,u0)[0,:]
    rk4sol2   = RungeKutta4(f,t2,u0)[0,:]
    rk4sol3   = RungeKutta4(f,t3,u0)[0,:]
    return t1, t2, t3, ex1, ex2, ex3, eulersol1, eulersol2, eulersol3, rk2sol1, rk2sol2, rk2sol3, rk4sol1, rk4sol2, rk4sol3

def plotting():
    t1,t2,t3,ex1,ex2,ex3,e1,e2,e3,r21,r22,r23,r41,r42,r43 = evaluation()
    plt.subplot(4, 3, 1)
    plt.plot(t1, e1)
    plt.title('Euler h = 0.1')
    plt.subplot(4, 3, 2) 
    plt.plot(t2, e2)
    plt.title('Euler h = 0.01')
    plt.subplot(4, 3, 3) 
    plt.plot(t3, e3)
    plt.title('Euler h = 0.001')
    plt.subplot(4, 3, 4)
    plt.plot(t1, r21)
    plt.title('RK2 h = 0.1')
    plt.subplot(4, 3, 5) 
    plt.plot(t2, r22)
    plt.title('RK2 h = 0.01')
    plt.subplot(4, 3, 6) 
    plt.plot(t3, r23)
    plt.title('RK2 h = 0.001')
    plt.subplot(4, 3, 7)
    plt.plot(t1, r41)
    plt.title('RK4 h = 0.1')
    plt.subplot(4, 3, 8) 
    plt.plot(t2, r42)
    plt.title('RK4 h = 0.01')
    plt.subplot(4, 3, 9) 
    plt.plot(t3, r43)
    plt.title('RK4 h = 0.001')
    plt.subplot(4, 3, 10)
    plt.plot(t1, ex1)
    plt.title('exact h = 0.1')
    plt.subplot(4, 3, 11) 
    plt.plot(t2, ex2)
    plt.title('exact h = 0.01')
    plt.subplot(4, 3, 12) 
    plt.plot(t3, ex3)
    plt.title('exact h = 0.001')
    plt.tight_layout()
    plt.show()

plotting()