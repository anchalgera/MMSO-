import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Task 12
# Crank Nicolson Scheme for solving parabolic PDEs u_t - kappa u_xx = f(t,x)

# --------------------------------------------------------------
# The PDE ------------------------------------------------------
# --------------------------------------------------------------

# Heat Equation u_t - kappa u_xx = f(t,x)
kappa = 1
def f(t, x):
    return (1 + np.pi**2 * t) * np.sin(np.pi * x)

# Initial Condition u(0,x)
def u0(x):
    return np.zeros_like(x)

# Zero Dirichlet Boundary Conditions: u(t,0) = u(t,L) = 0
def boundary_conditions(u_current):
    u_current[0]  = 0
    u_current[-1] = 0
    return u_current

# Reference Solution
def u_exact(t, x):
    return t * np.sin(np.pi * x)

# --------------------------------------------------------------
# Discretization -----------------------------------------------
# --------------------------------------------------------------

# Space Domain
h = 0.1
L = 2.0
n = int(L / h)
x = np.linspace(0, L, n)

# Time Domain
tau = 0.05
T = 1
m = int(T / tau)
t = np.linspace(0, T, m + 1)

# Grid Ratio
gamma = tau / h**2

# --------------------------------------------------------------
# Auxiliary functions ------------------------------------------
# --------------------------------------------------------------

def thomas_algorithm(A, b):
    '''
    Thomas Algorithm to solve Tridiagonal Matrices.
    A: Tridiagonal matrix
    b: Right hand side of Ax = b
    '''
    n = len(b)
    # Extract the sub, main and superdiagonal
    sub = np.diag(A, k=-1)
    mid = np.diag(A)
    sup = np.diag(A, k=1)
    c = np.zeros(n-1)
    d = np.zeros(n)
    # Forward elimination
    c[0] = sup[0] / mid[0]
    d[0] = b[0] / mid[0]
    for i in range(1, n-1):
        c[i] = sup[i] / (mid[i] - sub[i-1] * c[i-1])
    for i in range(1, n):
        d[i] = (b[i] - sub[i-1] * d[i-1]) / (mid[i] - sub[i-1] * c[i-1])
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n-2, -1, -1):
        x[i] = d[i] - c[i] * x[i+1]
    return x

# --------------------------------------------------------------
# Solving ------------------------------------------------------
# --------------------------------------------------------------

# Crank-Nicolson scheme
def crank_nicolson(u0, tau, n, m):
    # Constructing the tridiagonal matrix for implicit step
    A = np.diag(1 + kappa * gamma * np.ones(n)) - (kappa * gamma)/2 * np.diag(np.ones(n - 1), -1) - kappa * gamma/2 * np.diag(np.ones(n - 1), 1)
    B = np.diag(1 - kappa * gamma * np.ones(n)) + kappa * gamma/2 * np.diag(np.ones(n - 1), -1) + kappa * gamma/2 * np.diag(np.ones(n - 1), 1)
    u = np.zeros((m + 1, n))
    u[0] = u0
    for n in range(m):
        ti = n * tau
        b = np.dot(B, u[n]) + tau * f(ti, x)
        # Instead of inverting A to the other side we use thomas algorithm and solve A \ B+b
        u_next = thomas_algorithm(A,b)
        u[n+1] = boundary_conditions(u_next)
    return u

# --------------------------------------------------------------
# Plotting -----------------------------------------------------
# --------------------------------------------------------------

def surface_plotter(z):
    T_vals, X_vals = np.meshgrid(t, x)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_vals, T_vals, z.T, cmap='viridis')
    ax.set_title('Solution of u_t - u_xx = (1+pi^2*t)*sin(pi*x) with Crank-Nicolson scheme')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# --------------------------------------------------------------
# Output -------------------------------------------------------
# --------------------------------------------------------------

u_analytic = np.array([u_exact(ti,x) for ti in t])
u_solution = crank_nicolson(u0(x), tau, n, m)

surface_plotter(u_solution)