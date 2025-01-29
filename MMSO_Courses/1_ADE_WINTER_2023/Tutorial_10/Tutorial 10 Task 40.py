# Task 40 Sensitivities of logistic growth ODE

import numpy as np
import matplotlib.pyplot as plt

r = 0.1
k = 0.5
t0 = 0
u0 = 1

def solution(t):
    if u0 - k == 0:
        raise Exception("k cannot be the same as u0!")
    return k * (np.exp(np.log(u0/(u0-k))+k*r*t)) / (np.exp(np.log(u0/(u0-k))+k*r*t) -1)

def sensitivities(t):
    Sr = -(k**2 * t * u0 * (u0 - k) * np.exp(k*t*r)) / ((u0 * np.exp(k*t*r) - u0 + k)**2)
    Sk = ((u0 * np.exp(k*t*r)) * ((u0 * np.exp(k*t*r)) + r*t*k**2 - r*t*u0*k - u0)) / ((u0 * np.exp(k*t*r)) + k - u0)**2
    Su = (k**2 * np.exp(k*t*r)) / ((np.exp(k*t*r)-1)*u0 + k)**2
    return Sr, Sk, Su

def eval():
    t = np.linspace(0,100,10000)
    u = solution(t)
    Sr, Sk, Su = sensitivities(t)
    plt.subplot(1,4,1)
    plt.plot(t,u)
    plt.subplot(1,4,2)
    plt.plot(t,Sr)
    plt.subplot(1,4,3)
    plt.plot(t,Sk)
    plt.subplot(1,4,4)
    plt.plot(t,Su)
    plt.show()

eval()