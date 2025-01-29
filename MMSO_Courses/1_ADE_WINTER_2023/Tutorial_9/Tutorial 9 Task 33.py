import numpy as np
import matplotlib.pyplot as plt 



def ExponentialODE(t,u): #test
    return u

def ExponentialODEsolution(t):
    return np.exp(t)

def VanderPolODE(t,u):
    y1      = u[1]
    y1prime = 40 * (1- u[0]**2) * u[1] - u[0]
    return [y1, y1prime]


def ClassicalRungeKutta(f,t,u0):
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


def ExperimentalOrderofConsistency(h1,t0,tend,u,f,solver):
    h2      = h1/2
    tvector1 = np.linspace(t0,tend,int(1/h1))
    tvector2 = np.linspace(t0,tend,int(1/h2))
    u_exact1 = [u(t) for t in tvector1]
    u_exact2 = [u(t) for t in tvector2]
    u0       = [1.0, 0.0]
    u_sol1   = solver(f, tvector1, u0)
    u_sol2   = solver(f, tvector2, u0)
    delta1   = [abs(u_exact1[i] - u_sol1[0,:][i]) for i in range (len(u_exact1))]
    delta2   = [abs(u_exact2[i] - u_sol2[0,:][i]) for i in range (len(u_exact2))]
    error1   = np.max(delta1)  
    error2   = np.max(delta2)
    eoc_max  = np.log(error1/error2) / np.log(h1/h2)
    return eoc_max


def ExponentialTestCaseEvaluation():
    t0       = 0
    u0       = [1] 
    tend     = 10
    h        = 0.01
    t        = np.linspace(t0,10,int(1/h))
    u_exact  = [ExponentialODEsolution(ti) for ti in t]
    u_RK     = ClassicalRungeKutta(ExponentialODE, t, u0)
    error    = [np.abs(u_exact[i] - u_RK[0,i]) for i in range(len(u_exact))]
    eoc      = ExperimentalOrderofConsistency(h,t0,tend,ExponentialODEsolution,ExponentialODE,ClassicalRungeKutta)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(t, u_RK[0, :], color='lightgreen', label='Runge Kutta solution')
    axs[0].plot(t, u_exact, color='blue', label='Exact solution')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('u')
    axs[0].legend()
    axs[1].plot(t, error, color='red', label='error = |u_exact - u_RK|')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('|u_exact - u_RK|')
    axs[1].legend()
    eoctitle = "Experimental order of consistency = " + str(eoc)
    plt.title(eoctitle)
    plt.suptitle('Exponential ODE solution with classical Runge Kutta')
    plt.show()
    return u_exact, u_RK, error, eoc

def vanderPolEvaluation():
    t0       = 0
    u0       = [0.1, 0]
    h        = 0.00001 
    t        = np.linspace(t0, 100, int(1/h))
    u_RK     = ClassicalRungeKutta(VanderPolODE, t, u0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(t, u_RK[0, :], color='lightgreen', label='u (Runge Kutta)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('u')
    axs[0].legend()
    axs[1].plot(t, u_RK[1, :], color='blue', label='du/dt (Runge Kutta)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('du/dt')
    axs[1].legend()
    axs[2].plot(u_RK[0, :], u_RK[1, :], color='cyan', label='Phase space solution')
    axs[2].set_xlabel('u')
    axs[2].set_ylabel('du/dt')
    axs[2].legend()
    plt.suptitle('Van der Pol ODE solution')
    plt.tight_layout()
    plt.show()
    return u_RK

ExponentialTestCaseEvaluation()
vanderPolEvaluation()