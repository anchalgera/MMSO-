import numpy as np
import matplotlib.pyplot as plt
# Code for getting the transition probability of infections inside a household SIS model

beta  = np.random.rand() # transmission rate
gamma = np.random.rand() # recovery rate
I     = np.random.rand() # percentage of infected people
n     = 4                # people in the household

def factorial(n):
    res = 1
    while n > 1:
        res *= n
        n -= 1
    return res

def nchoosek(n,k):
    return factorial(n) / (factorial(n-k) * factorial(k))

def P(i,I):
    ''' 
    Probability of getting infected globally
    i: i infected individuals in a household
    I: Infected population percentage
    '''
    return 1 - (1-beta)**i * (1-beta*I)

def q(j,i):
    '''
    Probability of j agents getting infected
    j: new infected agents in a household
    i: already infected agents in a household
    '''
    return nchoosek(n-i,j) * P(i,I)**j * (1 - P(i,I))**((n-i)-j)

def r(k,i):
    '''
    Probability of k agents recovering
    k: agents recovering in the household
    i: already infected agents in a household
    '''
    return nchoosek(i,k) * gamma**k * (1-gamma)**(i-k)

def TransitionRate(i,j,k):
    '''
    Transition probabilities 
    i: already infected
    j: newly infected
    k: recovered
    '''
    return q(j,i) * r(k,i)

def MarkovMatrix(n):
    M = np.zeros((n,n))
    for row in range(n):
        for col in range(n):
            M[row,col] = TransitionRate(row,col,min(row,col))
    return M

def check_row_sums(matrix):
    """
    Check if each row of a matrix sums up to 1.
    """
    row_sums = np.sum(matrix, axis=1)
    return row_sums

M = MarkovMatrix(4)
print(f'beta = {beta}')
print(f'gamma = {gamma}')
print(f'I = {I}')
print(M)
print(check_row_sums(M))