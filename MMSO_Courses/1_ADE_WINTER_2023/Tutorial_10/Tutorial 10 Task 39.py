# Task 39 Stroemer Verlet Scheme

import numpy as np
import matplotlib.pyplot as plt

def StroemerVerlet(f,t,u0,uprime0):
    '''
    Stroemer-Verlet Scheme to solve second order autonomous ODEs.
    '''
    n = len(t)
    h = t[1] - t[0]
    uvector = [u0]
    u1 = u0 + h * uprime0 + (1/2) * h**2 * f(u0)
    uvector.append(u1)
    for i in range(1,n-1):
        unext = 2 * uvector[-1] - uvector[-2] + h**2 * f(uvector[-1])
        uvector.append(unext)
    return uvector

def uprimeprime(u):
    '''
    Second order autonomous ODE.
    '''
    return -np.sin(u)

def evaluation():
    t0 = 0
    t = np.linspace(t0,10,1000)
    u = StroemerVerlet(uprimeprime,t,1,0)
    plt.plot(t,u)
    plt.show()

evaluation()