
from scipy.optimize import fsolve
import scipy
import math
import numpy
from matplotlib import pyplot as plt


# Homework 1
# Terry Tianhao Wu


# Question 1

# define the parameters
gamma = 2
endowment1 = [1, 1, 2, 1, 2]
endowment2 = [1, 3, 1, 3, 1]
return1 = [1,1,1,1]
return2 = [1,1,1.5,1.5]
S = 4

# the unknowns to be solved first are: p1,p2,a11,a21,a12,a22


# define the nonlinear equations
def equations(p):
    p1,p2,a11,a21,a12,a22 = p
    return (a11+a12,a21+a22,-p1*float(endowment1[0]-p1*a11-p2*a21)**(-gamma)+sum(float(endowment1[i+1]+a11*return1[i]+a21*return2[i])**(-gamma)*return1[i] for i in [0,1,2,3])/S,-p2*float(endowment1[0]-p1*a11-p2*a21)**(-gamma)+sum(float(endowment1[i+1]+a11*return1[i]+a21*return2[i])**(-gamma)*return2[i] for i in [0,1,2,3])/S,-p1*float(endowment2[0]-p1*a12-p2*a22)**(-gamma)+sum(float(endowment2[i+1]+a12*return1[i]+a22*return2[i])**(-gamma)*return1[i] for i in [0,1,2,3])/S,-p2*float(endowment2[0]-p1*a12-p2*a22)**(-gamma)+sum(float(endowment2[i+1]+a12*return1[i]+a22*return2[i])**(-gamma)*return2[i] for i in [0,1,2,3])/S)
p1,p2,a11,a21,a12,a22 =  fsolve(equations, (1,1,0,0,0,0))


# Question 3

# define the utility function
def utility(c):
    return numpy.log(c)
# define the two production functions
def f1(k):
    return 0.9*k**(0.3)+0.3*k
def f2(k):
    return 1.1*k**(0.3)+0.9*k

# grid of values for state variable over which function will be approximated
grid = numpy.linspace(0.1,5,50)
grid = grid.tolist()

# define parameter
beta = 0.9

# define the expecation function
def w(m,v1,v2):
    inde = grid.index(m)
    return 0.5*(v1[inde]+v2[inde])

# define the bellman operator for the two value functions
def bellman1(k,w,v1,v2):
    # w is the expectation function
    # the maximum k'
    kmax = f1(k)+k
    vals = []
    for m in grid:
        # consumption is kmax-m
        # m is the saving
        if kmax-m>0:
            temp = utility(kmax-m) + beta * w(m,v1,v2)
        else:
            temp = -10000
        vals = vals + [temp]
    # the optimal saving
    a = vals.index(max(vals))
    b = max(vals)
    return [a,b]

# define the bellman operator for the two value functions
def bellman2(k,w,v1,v2):
    # w is the expectation function
    # the maximum k'
    kmax = f2(k)+k
    vals = []
    for m in grid:
        # consumption is kmax-m
        # m is the saving
        if kmax-m>0:
            temp = utility(kmax-m) + beta * w(m,v1,v2)
        else:
            temp = -10000
        vals = vals + [temp]
    # the optimal saving
    a = vals.index(max(vals))
    b = max(vals)
    return [a,b]

# initial guess
v1 = [1.1] * 50
v2 = [2.1] * 50

# updated guess
v1up = [3.1] * 50
v2up = [5.1] * 50

# define iteration parameters
tol, maxint = 0.0001,1000

# vector for policy
pol1 = v1[:]
pol2 = v2[:]

# loop for value iteration
timenum = 1
while (timenum<=maxint):
    for i in range(0,49):
        temp1 = bellman1(grid[i], w, v1, v2)
        temp2 = bellman2(grid[i], w, v1, v2)
        v1up[i]=temp1[1]
        v2up[i]=temp2[1]
        pol1[i]=grid[temp1[0]]
        pol2[i]=grid[temp2[0]]
    if (max([abs(a - b) for a, b in zip(v1, v1up)])>tol) | (max([abs(a - b) for a, b in zip(v2, v2up)])>tol):
        v1 = v1up[:]
        v2 = v2up[:]
        timenum = timenum+1
    else:break
    print(timenum)

# plot the policy function
plt.figure(1)
plt.plot(grid,pol1,grid,pol2,'-')

# then for 500 grid points

# grid of values for state variable over which function will be approximated
grid = numpy.linspace(0.1,10,500)
grid = grid.tolist()

# initial guess
v1 = [1.1] * 500
v2 = [2.1] * 500

# updated guess
v1up = [3.1] * 500
v2up = [5.1] * 500

# vector for policy
pol1 = v1[:]
pol2 = v2[:]

# loop for value iteration
timenum = 1
while (timenum<=maxint):
    for i in range(0,499):
        temp1 = bellman1(grid[i], w, v1, v2)
        temp2 = bellman2(grid[i], w, v1, v2)
        v1up[i]=temp1[1]
        v2up[i]=temp2[1]
        pol1[i]=grid[temp1[0]]
        pol2[i]=grid[temp2[0]]
    if (max([abs(a - b) for a, b in zip(v1, v1up)])>tol) | (max([abs(a - b) for a, b in zip(v2, v2up)])>tol):
        v1 = v1up[:]
        v2 = v2up[:]
        timenum = timenum+1
    else:break
    print(timenum)

# plot the policy function
plt.figure(2)
plt.plot(grid,pol1,grid,pol2,'-')

# Question 4

# define the utility function
def utility(c):
    return numpy.log(c)
# define the two production functions
def f1(k):
    return 0.9*k**(0.3)+0.3*k
def f2(k):
    return 1.1*k**(0.3)+0.9*k

# suppose v1 and v2 are the points in each piece
# the following function define the value of other capital values
# inputs are the function and k
# output is the value
def value(k,grid,v):
    resu = next(x[0] for x in enumerate(grid) if x[1] > k)
    resl = resu-1
    res = (k-grid[resl])/(grid[resu]-grid[resl])*(v[resu]-v[resl])
    return res

# define the expecation function
def w(m,v1,v2):
    # m is k'
    # v1 and v2 are the points defined
    res = 0.5*value(m,grid,v1)+0.5*value(m,grid,v2)
    return res

# define the parameter
beta = 0.9

# define the Bellman operator
# input the current capital level and shock
# return the policy and resulting value function
def bellman1(k,v1,v2):
    def f(x): return utility(f1(k)+k-x)+beta*w(x,v1,v2)
    temp = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=[min(grid),max(grid)], method='bounded')
    return [temp.x,temp.fun]

def bellman2(k,v1,v2):
    def f(x): return utility(f2(k)+k-x)+beta*w(x,v1,v2)
    temp = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=[min(grid),max(grid)], method='bounded')
    return [temp.x,temp.fun]

# define the grid, maximum iteration steps, and tolreance
grid = numpy.linspace(10,100,29)
grid = grid.tolist()
tol, maxint = 0.01,2000

# guess the initial policy function
pol1=[1.3]*29
pol2=[2.3]*29

# calculate the intital value using the policy function
value1=[1.1]*29
value2=[1.1]*29

# define the update policy function
polup1=[3.3]*29
polup2=[3.3]*29

# loop for time iteration
timenum = 1
while (timenum<=maxint):
    # calculate the update policy for each point
    for i in range(0, 28):
        temp = bellman1(grid[i], value1, value2)
        polup1[i] = temp[0]
        value1[i] = temp[1]

        temp = bellman2(grid[i], value1, value2)
        polup2[i] = temp[0]
        value2[i] = temp[1]
    # stopping rule
    if (max([abs(a - b) for a, b in zip(pol1, polup1)])>tol) | (max([abs(a - b) for a, b in zip(pol2, polup2)])>tol):
        pol1 = polup1[:]
        pol2 = polup2[:]
        timenum = timenum+1
    else:break
    print(timenum)


# plot the policy function
plt.figure(3)
plt.plot(grid,pol1,grid,pol2,'-')

# new v(c) and beta
beta = 0.999

# define the utility function
def utility(c):
    return -c**(-4)

# define the Bellman operator
# input the current capital level and shock
# return the policy and resulting value function
def bellman1(k,v1,v2):
    def f(x): return utility(f1(k)+k-x)+beta*w(x,v1,v2)
    temp = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=[min(grid),max(grid)], method='bounded')
    return [temp.x,temp.fun]

def bellman2(k,v1,v2):
    def f(x): return utility(f2(k)+k-x)+beta*w(x,v1,v2)
    temp = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=[min(grid),max(grid)], method='bounded')
    return [temp.x,temp.fun]

# define the grid, maximum iteration steps, and tolreance
grid = numpy.linspace(10,100,29)
grid = grid.tolist()
tol, maxint = 0.01,2000

# guess the initial policy function
pol1=[1.3]*29
pol2=[2.3]*29

# guess the initial value function
value1=[0.3]*29
value2=[0.3]*29

# define the update policy function
polup1=[3.3]*29
polup2=[3.3]*29

# loop for time iteration
timenum = 1
while (timenum<=maxint):
    # calculate the update policy for each point
    for i in range(0, 28):
        temp = bellman1(grid[i], value1, value2)
        polup1[i] = temp[0]
        value1[i] = temp[1]

        temp = bellman2(grid[i], value1, value2)
        polup2[i] = temp[0]
        value2[i] = temp[1]
    # stopping rule
    if (max([abs(a - b) for a, b in zip(pol1, polup1)])>tol) | (max([abs(a - b) for a, b in zip(pol2, polup2)])>tol):
        pol1 = polup1[:]
        pol2 = polup2[:]
        timenum = timenum+1
    else:break
    print(timenum)


# plot the policy function
plt.figure(4)
plt.plot(grid,pol1,grid,pol2,'-')


# define the parameter
beta = 0.9

# the iteration parameters
tol, maxint = 0.0001,1e6

# define the interpolation
def ip(x, k1,k2):
    return numpy.interp(x,k1,k2)

# define the utility function
def utility(c):
    return numpy.log(c)

# define the derivative
def utilityd(c):
    return 1/c

# define the two production functions
def f(k,s):
    if s==1:
        return 0.9*k**(0.3)+0.3*k
    if s==2:
        return 1.1*k**(0.3)+0.9*k

# define the derivate of the production functions
def fd(k,s):
    if s==1:
        return 0.9*0.3*k**(0.3-1)+0.3
    if s==2:
        return 1.1*0.3*k**(0.3-1)+0.9

# define the capital space
grid = numpy.linspace(0.1,20,350)
grid = grid.tolist()

# define the Euler equation
def foc(x,f,fd,utilityd,k2,k):
    k1s = k2[:350]
    k2s = k2[350:]
    y1 = x[:350]
    y2 = x[350:]
    return numpy.append(utilityd(f(k, 1) - y1) - beta * (0.5 * fd(y1, 1) * utilityd(f(y1, 1) - ip(y1, k, k1s)) + 0.5 * fd(y1, 2) * utilityd(f(y1, 2) - ip(y1, k, k1s))),
                     utilityd(f(k, 2) - y2) - beta * (0.5 * fd(y2, 1) * utilityd(f(y2, 1) - ip(y2, k, k2s)) + 0.5 * fd(y2, 2) * utilityd(f(y2, 2) - ip(y2, k, k2s))))

# iterations
old_cap = numpy.ones(2 * len(grid))
iter = 0
while True:
    iter = iter + 1
    pr = f,fd,utilityd,old_cap,grid
    solve = scipy.optimize.root(foc, old_cap, method='hybr', args=(pr))
    new_k = solve.x
    error = numpy.linalg.norm(new_k - old_cap)
    if error <= tol: break
    old_cap = new_k
