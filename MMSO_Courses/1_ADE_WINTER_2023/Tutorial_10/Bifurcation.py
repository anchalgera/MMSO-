#logistic growth bifurcation diagram

import matplotlib.pyplot as plt
import numpy as np
P=np.linspace(0.7,4,10000)
m=0.7
X = []
Y = []
for u in P:
    X.append(u)
    # Start with a random value of m instead of remaining stuck
    # on a particular branch of the diagram
    m = np.random.random()
    for n in range(1001):
      m=(u*m)*(1-m) 
    for l in range(1051):
      m=(u*m)*(1-m)
    # Collection of data in Y must be done once per value of u
    Y.append(m)
plt.plot(X, Y, ls='', marker=',')
plt.show()