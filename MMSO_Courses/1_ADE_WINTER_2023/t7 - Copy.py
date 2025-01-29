import matplotlib.pyplot as plt
def u(t):
 return (1 / ((1 + 24 * t**2) ** (1/2)))

def u_derivative(t):
 return (-24 * t) / ((1 + 24 * t ** 2) ** (3/2))

def evaluate_function(tdata):
 udata = []
 for t in tdata:
    udata.append(u(t))
 return udata

def evaluate_derivative(tdata):
 udata = []
 for t in tdata:
    udata.append(u_derivative(t))
 return udata

def evaluate_u_left(tdata):
 n = len(tdata)
 h = 1/n
 udata = []
 udata.append(u(tdata[0]))
 for i in range(1,n):
    udata.append((u(tdata[i]) - u(tdata[i-1]))/h)
 return udata

def evaluate_u_center(tdata):
 n = len(tdata)
 h = 1/n
 udata = []
 udata.append(u(tdata[0]))
 for i in range(1, n-1):
    udata.append((u(tdata[i+1]) - u(tdata[i-1]))/(2*h))
 udata.append(u(tdata[-1])) #-1 accesses the last entry of a vector
 return udata

def evaluate_u_right(tdata):
 n = len(tdata)
 h = 1/n
 udata = []
 udata.append(u(tdata[0]))
 for i in range(0,n-1):
    udata.append((u(tdata[i+1]) - u(tdata[i]))/h)
 return udata

n = 100
tdata = [j/n for j in range(n+1)]
correct = evaluate_function(tdata)
derivative = evaluate_derivative(tdata)
leftsided = evaluate_u_left(tdata)
centered = evaluate_u_center(tdata)
rightsided = evaluate_u_right(tdata)

plt.plot(tdata,derivative,linestyle='-',label='derivative')
plt.plot(tdata,leftsided,linestyle='--',label='left')
plt.plot(tdata,centered,linestyle='-.',label='center')
plt.plot(tdata,rightsided,linestyle=':',label='right')
plt.legend()
plt.show()