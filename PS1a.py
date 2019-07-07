
import sympy as sy
from sympy import *
import numpy as np
from matplotlib import pyplot as plt
from autograd import grad
from autograd import numpy as anp
from autograd import elementwise_grad
import random as rand
import scipy.stats as sts
from scipy import optimize as opt
from scipy import integrate

# Differentiation

# 1
# define the function
def myfun(x):
  return (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
# take derivative
x = sy.symbols('x')
print(sy.diff(myfun(x),x))
# define the derivative function
def mydev(x):
  return sy.diff(myfun(x),x)
# lambdify the function
lam_f = lambdify(x, myfun(x))
lam_v = lambdify(x, mydev(x))
# plot
plt.figure(1)
t1 = np.arange(-4, 4, 0.005)
plt.plot(t1, lam_f(t1),t1, lam_v(t1))
ax = plt.gca()
ax.spines["bottom"].set_position("zero")

#2
# define the approximation
def forward1(x,h,f):
  return (f(x+h)-f(x))/h
def forward2(x,h,f):
  return (-3*f(x)+4*f(x+h)-f(x+2*h))/2/h
def backward1(x,h,f):
  return (f(x)-f(x-h))/h
def backward2(x,h,f):
  return (3*f(x)-4*f(x-h)+f(x-2*h))/2/h
def centered1(x,h,f):
  return (f(x+h)-f(x-h))/2/h
def centered2(x,h,f):
  return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/12/h

# lambdify the function
lam_1 = lambdify(x, forward1(x,0.01,myfun))
lam_2 = lambdify(x, forward2(x,0.01,myfun))
lam_3 = lambdify(x, backward1(x,0.01,myfun))
lam_4 = lambdify(x, backward2(x,0.01,myfun))
lam_5 = lambdify(x, centered1(x,0.01,myfun))
lam_6 = lambdify(x, centered2(x,0.01,myfun))

# plot
plt.figure(2)
plt.plot(t1,lam_v(t1),'-' ,label="Analytical")
plt.plot(t1,lam_1(t1), '.',label="Forward Approximation1")
plt.plot(t1,lam_2(t1), ':',label="Forward Approximation2")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.figure(3)
plt.plot(t1,lam_v(t1),'-' ,label="Analytical")
plt.plot(t1,lam_3(t1), '.',label="Backward Approximation1")
plt.plot(t1,lam_4(t1), ':',label="Backward Approximation2")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.figure(4)
plt.plot(t1,lam_v(t1),'-' ,label="Analytical")
plt.plot(t1,lam_5(t1), '.',label="Centered Approximation1")
plt.plot(t1,lam_6(t1), ':',label="Centered Approximation2")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

#3
# define the difference function at a x and a h for each approximation
def diff1(x,h):
  return  abs(forward1(x,h,myfun)-lam_v(x))
def diff2(x,h):
  return  abs(forward2(x,h,myfun)-lam_v(x))
def diff3(x,h):
  return  abs(backward1(x,h,myfun)-lam_v(x))
def diff4(x,h):
  return  abs(backward2(x,h,myfun)-lam_v(x))
def diff5(x,h):
  return  abs(centered1(x,h,myfun)-lam_v(x))
def diff6(x,h):
  return  abs(centered2(x,h,myfun)-lam_v(x))

# lambdify the function
lam_1 = lambdify(x, diff1(1,x))
lam_2 = lambdify(x, diff2(1,x))
lam_3 = lambdify(x, diff3(1,x))
lam_4 = lambdify(x, diff4(1,x))
lam_5 = lambdify(x, diff5(1,x))
lam_6 = lambdify(x, diff6(1,x))

# plot
t1 = np.logspace(-8, 0, 9)
plt.figure(5)
plt.loglog(t1,lam_1(t1),label="Forward Approximation1")
plt.loglog(t1,lam_2(t1),label="Forward Approximation2")
plt.loglog(t1,lam_3(t1),label="Backward Approximation1")
plt.loglog(t1,lam_4(t1),label="Backward Approximation2")
plt.loglog(t1,lam_5(t1),label="Centered Approximation1")
plt.loglog(t1,lam_6(t1),label="Centered Approximation2")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


#5
# define the function
def myfun(x):
  return sum(x)

# define the derivative approximation
def mydev(x,h,j,myfun):
  f1 = myfun(np.append(np.append(np.zeros(j-1),1),np.zeros(len(x)-j))*h+x)
  f2 = myfun(x-np.append(np.append(np.zeros(j-1),1),np.zeros(len(x)-j))*h)
  return [(x - y)/2/h for x, y in zip(f1, f2)]
# try the function
def tryfun(x):
  return [x[0]**2,x[0]**3-x[1]]

mydev([1,1],0.1,1,tryfun)

#6
# define the function
def myfun(x,n):
  res = 0
  if n==0: res=1
  if n==1: res=x
  if n==2: res=[1,x,2*x*x-1]
  if n>=3:
    init=[1,x,2*x*x-1]
    for i in range(2, n):init = np.append(init,2*x*init[i-1]-init[i-2])
  if n>=2: res=init[n]
  return res

# derivative
g = lambda x: myfun(x,3)
dg = elementwise_grad(g)
# plot
t1 = anp.arange(-1, 1, 0.01,dtype=anp.float)
plt.figure(6)
plt.plot(t1,dg(t1))

# Newton's Method

#1
# define the newton method
def myfun(x_0,tol,max,f,fdev):
  i = 1
  while abs(f(x_0))>tol:
    x_0 = x_0-(float(f(x_0))/fdev(x_0))
    i = i + 1
    print(i)
    if (i > max):
      break
  return [x_0,abs(f(x_0))<=tol]

# try the function
def tryfun(x):
  return x**4-3
def trydev(x):
  return 4*x**3

# run the method
myfun(2,0.0001,10**5,tryfun,trydev)


#3
# define the newton method
def myfun(x_0,tol,max,f,fdev,a):
  i = 1
  while abs(f(x_0))>tol:
    x_0 = x_0-a*(f(x_0)/fdev(x_0))
    i = i + 1
    print(i)
    if (i > max):
      break
  return [x_0,abs(f(x_0))<=tol]

# try the function
f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
fdev = lambda x: 1./3*np.sign(x) * np.power(np.abs(x), -2./3)

# run the method
myfun(0.01,0.05,10**5,f,fdev,1)
myfun(0.01,0.05,10**5,f,fdev,0.4)

#4
# define the newton method
def myfun(x_0,tol,max,a):
  i = 1
  while abs(f(x_0))>tol:
    x_0 = x_0-a*3*x_0
    i = i + 1
    #print(i)
    if (i > max):
      break
  return i

# define a vector
t1 = [x * 0.01 for x in range(1, 100)]
res = []

# run
k= 0
for i in t1:
  res = res + [myfun(0.01, 0.0001, 10 ** 4, i)]
  k = k + 1
  print(k)

# plot
plt.figure(6)
plt.plot(t1,res,'-')


#7
def ex7func(f, fprime, zeros, rmin, rmax, imin, imax, res, iters=10):

  rgrid = np.linspace(rmin, rmax, res)
  igrid = np.linspace(imin, imax, res)
  Xreal, Ximag = np.meshgrid(rgrid, igrid)
  X_0 = Xreal + Ximag * 1j


  X_t = X_0
  for i in range(iters):
    X_t = X_t - f(X_t) / fprime(X_t)
    i += 1


  Y = np.zeros_like(X_t, dtype=int)
  for i in range(res):
    for j in range(res):
      Y[i, j] = np.argmin(np.abs(zeros - X_t[i, j]))

  plt.pcolormesh(Xreal, Ximag, Y,cmap="brg")
  plt.show()

f7a = lambda x: x ** 3 - 1
f7b = lambda x: x ** 3 - x
fp7a = lambda x: 3 * x ** 2
fp7b = lambda x: 3 * x ** 2 - 1
f7zerosa = [-1 / 2 - np.sqrt(3) / 2 * 1j, -1 / 2 + np.sqrt(3) / 2 * 1j, 1]
f7zerosb = [0, -1, 1]
ex7func(f7a, fp7a, f7zerosa, -1.5, 1.5, -1.5, 1.5, 500, 50)
ex7func(f7b, fp7b, f7zerosb, -1.5, 1.5, -1.5, 1.5, 500, 50)



# Numerical Integration

#2.1
def ex1func(g, a, b, N, method='trapezoid'):
  # Partition boundary
  partition = np.linspace(a, b, N)
  # initialize integral
  integral = 0
  if method == 'midpoint':
    for i in range(N - 1):
      midpoint = (partition[i] + partition[i + 1]) / 2
      integral += g(midpoint)
    integral = integral * ((b - a) / N)
  elif method == 'trapezoid':
    for i in range(1, N - 1):
      integral += 2 * g(partition[i])
    integral += g(partition[0]) + g(partition[N - 1])
    integral = integral * (b - a) / (2 * N)
  elif method == 'Simpsons':
    for i in range(N - 1):
      midpoint = (partition[i] + partition[i + 1]) / 2
      integral += ((partition[i + 1] - partition[i]) / 6) * \
                  (g(partition[i]) + 4 * g(midpoint) + g(partition[i + 1]))
  else:
    raise ValueError('method is invalid')
  return integral

#evaluating
g = lambda x: 0.1*x**4 - 1.5*x**3 + 0.53*x**2 + 2*x + 1
ex1func(g, -1, 1, 100, method='midpoint')
ex1func(g, -1, 1, 100, method='trapezoid')
ex1func(g, -1, 1, 100, method='Simpsons')


#2.2
def normalapprox(mu, sigma, N, k):
  # define left and right bounds
  lbound = mu - sigma * k
  rbound = mu + sigma * k
  Zvec = np.linspace(lbound, rbound, N)
  # initialize weight vector
  Wvec = np.zeros_like(Zvec)
  for i in range(N):
    if i == 0:
      Wvec[i] = sts.norm.cdf((Zvec[i] + Zvec[i + 1]) / 2, \
                             loc=mu, scale=sigma)
    elif i == N - 1:
      Wvec[i] = 1 - sts.norm.cdf((Zvec[i - 1] + Zvec[i]) / 2, \
                                 loc=mu, scale=sigma)
    else:
      # Using midpoint method
      Wvec[i] = sts.norm.pdf(Zvec[i], loc=mu, scale=sigma) \
                * (rbound - lbound) / N
  return Zvec, Wvec

#try the function
Zvec, Wvec = normalapprox(0, 1, 11, 4)
plt.plot(Zvec, Wvec, '-')
plt.xlabel('x')
plt.ylabel('f(x)')

#2.3
def lognormalapprox(mu, sigma, N, k):
  Zvecnorm, Wvecnorm = normalapprox(mu, sigma, N, k)
  # Transform node location while keeping weights same
  Zvec = np.exp(Zvecnorm)
  Wvec = Wvecnorm
  return Zvec, Wvec

#plot
Zvec, Wvec = lognormalapprox(1, .5, 35, 4)
plt.plot(Zvec, Wvec, '-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

#2.4
def expinc(mu, sigma, N, k):
  Zvec, Wvec = lognormalapprox(mu, sigma, N, k)
  meaninc = np.average(Zvec, weights=Wvec)
  return meaninc

# evaulation
mu = 3.5
sigma = .2
N = 500
k = 4
estval = expinc(mu, sigma, N, k)
trueval = np.exp(mu + (sigma**2)/2)
# the difference
print(abs(estval-trueval))

#3.1
def system_errors(x):
  w1, w2, w3, x1, x2, x3 = x
  rhs = np.zeros(6, dtype=np.float64)
  for ii in range(6):
    rhs[ii] = (1 ** (ii + 1) - (-1) ** (ii + 1)) / (ii + 1)
  err = np.zeros(6, dtype=np.float64)
  for ii in range(6):
    err[ii] = np.abs(rhs[ii] - w1 * (x1 ** ii) - w2 * (x2 ** ii) - w3 * (x3 ** ii))
  return err

guess = np.array([0.6, 1.3, -1, -.8, 0, .2])
sol = opt.fsolve(system_errors, guess)
print(sol)
#define the function
def g(x):
  return .1 * x ** 4 - 1.5 * x ** 3 + .53 * x ** 2 + 2 * x + 1
np.sum(sol[:3] * g(sol[3:]))

#3.2
vv, err = integrate.quad(g, -10, 10)
print(vv)
print(err)

#4.1
rand.seed(25)
def mc(g, x, omega, N):
  n = len(x)
  vol = np.prod(omega[:, 1] - omega[:, 0])
  integral = 0
  for ii in range(N):
    draw = np.random.rand(n) * (omega[:, 1] - omega[:, 0]) + omega[:, 0]
    integral += g(draw.T)
  integral = vol * (1 / N) * integral

  return integral

  def g(x):
    if x[0] ** 2 + x[1] ** 2 <= 1:
      val = 1
    else:
      val = 0
    return val

# approximate pi
omega = np.array([[-1, 1], [-1, 1]])
a1=mc(g, np.array([0,0]), omega, 10000)
a2=mc(g, np.array([0,0]), omega, 1000000)
print(a1)
print(a2)

