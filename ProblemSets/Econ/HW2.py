
from scipy.optimize import fsolve
import scipy
import math
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Homework 2
# Terry Tianhao Wu


# Question 5

# define the parameters
gamma = 2.5
beta = 0.98
alpha = 0.4
delta = 0.1
tau = 0.05

# define the nonlinear equations
def equations(p):
    c,w,r,K,T = p
    return (   (w+(r-delta)*K)*(1-tau)+T-c, beta*((r-delta)*(1-tau)+1)-1, r-alpha*K**(alpha-1), w-K**alpha*(1-alpha), T/tau-(w+(r-delta)*K)   )

c,w,r,K,T=  fsolve(equations, (1, 1, 1, 1, 1))

# analytical solution
KK = (1/beta-1)/alpha/(1-tau)+delta/alpha
KK = KK ** (1/(alpha-1))


# Question 6

# define the parameters
gamma = 2.5
xi = 1.5
beta = 0.98
alpha = 0.4
a = 0.5
delta = 0.1
tau = 0.05

# define the nonlinear equations
def equations(p):
    c,w,r,K,T,L = p
    return ((w*L+(r-delta)*K)*(1-tau)+T-c, beta*((r-delta)*(1-tau)+1)-1, r-alpha*K**(alpha-1)*L**(1-alpha), w-K**alpha*(1-alpha)*L**(-alpha), T/tau-(w*L+(r-delta)*K), a*(1-L)**(-xi)-w*(1-tau)*c**(-gamma))

c,w,r,K,T,L= fsolve(equations, (1, 2.34, 1, 1, 1, 0.1))


# PS2 Exercise 1

# define the parameters
beta = 0.98
alpha = 0.4

# steady states
A = alpha * beta
kss = A**(1/(1-alpha))

# define the matrices
F = (alpha*kss**(alpha-1))/(kss**alpha-kss)
G = -(alpha*kss**(alpha-1)*(alpha+kss**(alpha-1)))/(kss**alpha-kss)
H = (alpha**2*kss**(2*alpha-2))/(kss**alpha-kss)
L = -(alpha*kss**(2*alpha-1))/(kss**alpha-kss)
M = (alpha**2*kss**(2*alpha-2))/(kss**alpha-kss)
P = ((G**2-4*F*H)**(1/2)-G)/2/F
Q = - (L+M)/(F+F*P+G)

# define the function
def my(k,z):
    return kss+P*(k-kss)+Q*z

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
kdata = np.arange(-5, 5, 0.1)
zdata = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(kdata, zdata)
Z = my(X,Y)

# plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# PS3 Exercise 2

# parameters
alpha = 0.33
k = 5
z = 1
b = 2
t = 0.1
h = 24

# define the nonlinear equations
def equations(p):
    pi, w = p
    nd =  ((1-alpha)*z/w)**(1/alpha)*k
    return ( nd - h + (w*h+pi-t) * b/w/(1+b) , z*k**alpha*nd**(1-alpha)-w*nd-pi )

pi,w= fsolve(equations, (0.4, 0.7))
