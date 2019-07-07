
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
from itertools import chain, combinations
import sys
import bx
from scipy import stats
from random import choice
import random
import scipy as sp

# Introduction to NumPy

#1
A = np.array([[3,-1,4],[1,5,-9]])
B = np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
print(A @ B)

#2
A = np.array([[3,1,4],[1,5,9],[-5,3,1]])
print(-A @ A @ A + 9*A@A-15*A)

#3
samp = np.ones(7)
A= np.triu(samp, k=0)

samp = np.array([5]*7)
B= np.triu(samp, k=0)
samp=np.array([-1]*7)
C=np.triu(samp, k=1)
C=np.transpose(C)
B = B + C
# matrix ABA
res = A @ B @ A
res = np.float64(res)

#4
#define function
def myfun(x):
  temp = np.abs(x)
  temp = temp + x
  temp = temp/2
  return temp
# try it
myfun(np.array([1,3,-1,-5]))

#5
A = np.array([[0,2,4],[1,3,5]])
B = np.array([[3,0,0],[3,3,0],[3,3,3]])
C = np.array([[-2,0,0],[0,-2,0],[0,0,-2]])
D = np.eye(3)
row1 = np.column_stack((np.zeros_like(D), A.T, D))
row2 = np.hstack((A, np.zeros((2,2)) , np.zeros_like(A)))
row3 = np.hstack( (B, np.zeros_like(A.T), C))
result = np.vstack((row1, row2, row3))
result

#6
def myfun(mat):
    i,j = np.shape(mat)
    sums = mat.sum(axis=1).reshape((j,1))
    result = mat / sums
    return result
# testing
test = np.arange(16).reshape((4,4))
print(myfun(test))

#7
grid = np.load("grid.npy")
# define my function
def myfun():
  hmax = np.max(grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:])
  vmax = np.max(grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :])
  drmax = np.max(grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:])
  lrmax = np.max(grid[3:, :-3] * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:])
  a = max(hmax,vmax,drmax,lrmax)
  return a

# Standard Library

#1
def myfun(x):
  return [min(x),max(x),sum(x)/len(x)]

#2
def myfun(object1):
    object2 = object1
    if type(object1) is list:
      object2.append(1)
    elif type(object1) is dict:
      object2[1] = 'a'
    elif type(object1) is tuple:
      object2 += (1,)
    elif type(object1) is str:
      object2 += 'a'
    else:
      object2 += 1
    Statement = object2 == object1
    return print(str(type(object1)) + " is mutable: " + str(Statement))
my_l = []
my_dict = {}
my_num = 0
my_str = ''
my_tuple = ()
# results
myfun(my_l)
myfun(my_dict)
myfun(my_str)
myfun(my_num)
myfun(my_tuple)

#3
import Calculator as ca

def myfun(a, b):
    res = ca.sqrt(ca.sum(ca.prod(a,a), ca.prod(b,b)))
    return res
side1 = 3
side2 = 4
myfun(side1, side2)

#4
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def power(items):
    return [list(i) for i in powerset(items)]
test = (1,9,8,10)
print(power(test))

#5
def get_name():
    if len(sys.argv) != 2:
        name = input("Tell me your name")
    else:
        name = sys.argv[1]
    return name


def rolldice(remaining):
    a = random.randint(1, 6)
    b = random.randint(1, 6)
    if score(remaining) > 6:
        return a + b
    else:
        return a

def score(a):
    return sum(a)

def game_over(remaining):
    result = score(remaining)
    print("Score for player " + name + " : " + str(result) + " points")
    if result == 0:
        print("You shut the box.")
    return None

name = get_name()
remaining = [1, 2, 3, 4, 5, 6, 7, 8, 9]
while True:
    if score(remaining) != 0:
        roll = rolldice(remaining)
        print("\n")
        print("Numbers left: " + str(remaining))
        print("Roll: " + str(roll))
        if bx.isvalid(roll, remaining):
            while True:
                elim = input("Which numbers to flip: ")
                choice = bx.parse_input(elim, remaining)
                if len(choice) > 0:
                    if sum(choice) == roll:
                        break
                    else:
                        print("Invalid Input")
                else:
                    print("Invalid Input")
            remaining = [num for num in remaining if num not in choice]
        else:
            print("Game Over.")
            print("\n")
            game_over(remaining)
            break
    else:
        game_over(remaining)
        break

# Visualization

#2
b_pol = lambda v,n,x :sp.special.binom(n,v) * (x**v) * ((1-x)**(n-v))
x_dom = np.linspace(1e-10,1+1e-10,100)
for i in range(3): # for n
    for j in range(3): # for v
        m = i*j + j
        plt.subplot(i+1,j+1,m+1)
        plt.plot(x_dom, b_pol(j,i,x_dom))
        plt.xlim(0,1)
        plt.ylim(0,1)

#3
data_MLB = np.load('MLB.npy')
#plot
result_MLB1  = stats.linregress(data_MLB[:,0], data_MLB[:,2])
plt.scatter(data_MLB[:,0], data_MLB[:,2],alpha=0.5,linewidth =1.0 ,edgecolor='g')
plt.plot(data_MLB[:,0], result_MLB1[1]+result_MLB1[0]*data_MLB[:,0])
plt.xlabel('Height')
plt.ylabel('Age')
plt.show()
#plot
result_MLB2  = stats.linregress(data_MLB[:,1], data_MLB[:,2])
plt.scatter(data_MLB[:,1], data_MLB[:,2],alpha=0.5,linewidth =1.0 ,edgecolor='k')
plt.plot(data_MLB[:,1], result_MLB2[1]+result_MLB2[0]*data_MLB[:,1])
plt.xlabel('Weight')
plt.ylabel('Age')
plt.show()

#5
x = np.linspace(0.0, 2.0, 200)
X, Y = np.meshgrid(x, x)
Z = (1-X)**2 + 100*((Y-X**2)**2)
plt.pcolormesh(X, Y, Z, cmap="viridis") # Heat map.
plt.scatter(1, 1, alpha=1.0,linewidth =3.0 ,edgecolor='r')
plt.figtext(.5, .5, "Minimum")
plt.show()

plt.contour(X, Y, Z, 100, cmap="viridis") # Contour map.
plt.scatter(1, 1, alpha=1.0,linewidth =3.0 ,edgecolor='r')
plt.figtext(.5, .5, "Minimum")
plt.show()

#6
countries = ["Austria", "Bolivia", "Brazil", "China","Finland", "Germany", "Hungary", "India","Japan", "North Korea", "Montenegro", "Norway","Peru", "South Korea", "Sri Lanka", "Switzerland","Turkey", "United Kingdom", "United States", "Vietnam"]
data_coun = np.load('countries.npy')

plt.scatter(data_coun[:,0], data_coun[:,1])
plt.xlabel('Population')
plt.ylabel('GDP')
plt.show()

plt.hist(data_coun[:,0], bins=20)
plt.xlabel('GDP')
plt.ylabel('Frequency')
plt.show()

plt.barh(countries, data_coun[:,2], align='center', alpha=1.0)
plt.xlabel('Average male Height')
plt.show()

plt.barh(countries, data_coun[:,3], align='center', alpha=1.0)
plt.xlabel('Average female Height')
plt.show()


#Matplotlib

#1
def find_variance(n):
    std_nm_array = np.random.normal(size=(n,n))
    means = np.mean(std_nm_array, axis = 1)
    return np.var(means)
def plot_var():
    x = np.arange(100, 1000, 100)
    plt.plot(np.array([find_variance(i) for i in x]))
    plt.show()
plot_var()
plt.ion()
x = np.linspace(1, 4, 100)
plt.plot(x, np.log(x))
plt.plot(x, np.exp(x))

#2
x = np.linspace(- 2 * np.pi, 2 * np.pi, 100)
cosy = np.cos(x)
siny = np.sin(x)
arctany = np.arctan(x)
plt.plot(x, cosy)
plt.plot(x, siny)
plt.plot(x, arctany)
plt.show()

#3
x1 = np.linspace(-2, 1, 100)
x1 =  x1[:-1]
x2 = np.linspace(1, 6, 100)
x2 = x2[1:]
y1 = 1 / (x1 - 1)
y2 = 1 / (x2 - 1)
plt.plot(x1, y1, 'm--', lw=4)
plt.plot(x2, y2, 'm--', lw=4)
plt.xlim(-2, 6)
plt.ylim(-6, 6)
plt.show()

#4
x = np.linspace(0, 2*np.pi, 100)
fig , axes = plt.subplots(2, 2)
axes[0][0].plot(x, np.sin(x), 'g-')
axes[1][0].plot(x, np.sin(2*x), 'r--')
axes[0][1].plot(x, 2*np.sin(x), 'b--')
axes[1][1].plot(x, 2*np.sin(2*x), 'm:')
axes[0][0].axis([0, 2*np.pi, -2, 2])
axes[1][0].axis([0, 2*np.pi, -2, 2])
axes[0][1].axis([0, 2*np.pi, -2, 2])
axes[1][1].axis([0, 2*np.pi, -2, 2])
axes[0][0].set_title("sin(x)")
axes[1][0].set_title("sin(2x)")
axes[0][1].set_title("2sin(x)")
axes[1][1].set_title("2sin(2x)")
fig.suptitle("Variations of sin(x)")
plt.tight_layout()
plt.show()

#5
crash_data = np.load('FARS.npy')
fig , axes = plt.subplots(1,2)
axes[0].plot(crash_data[:, 1], crash_data[:, 2], 'k,')
axes[0].set_aspect("equal")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[1].hist(crash_data[:,0], bins=24)
axes[1].axis([0, 24, 0, 9000])
plt.tight_layout()
plt.show()

# OOP

#1
class Backpack:
    def __init__(self, name, color, max_size=5):  # This function is the constructor.
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        if len(self.contents) >= self.max_size:
            print("No Room!")
            return
        else:
            self.contents.append(item)  # Use 'self.contents', not just 'contents'.

    def take(self, item):
        self.contents.remove(item)

    def dump(self):
        self.contents = []
def test_backpack():
    testpack = Backpack("Barry", "black")       # Instantiate the object.
    if testpack.name != "Barry":                # Test an attribute.
        print("Backpack.name assigned incorrectly")
    for item in ["pencil", "pen", "paper", "computer"]:
        testpack.put(item)                      # Test a method.
    print("Contents:", testpack.contents)
    testpack.take("pencil")
    print("Contents:", testpack.contents)
    testpack.dump()
    print("Contents:", testpack.contents)
test_backpack()

#2
class Jetpack(Backpack):
    def __init__(self, name, color, max_size=2, fuel_amount=10):
        Backpack.__init__(self, name, color, max_size)
        self.fuel_amount = fuel_amount

    def put(self, item):
        if len(self.contents) >= self.max_size:
            print("No Room!")
            return
        else:
            self.contents.append(item)  # Use 'self.contents', not just 'contents'.

    def take(self, item):
        self.contents.remove(item)

    def fly(self, fuel_burned):
        if fuel_burned > self.fuel_amount:
            print("Not enough fuel!")
            return
        else:
            self.fuel_amount = fuel_burned

    def dump(self):
        self.fuel_amount = []
        self.contents = []

#3
class Backpack:
    """A Backpack object class. Has a name, color, maximum size, and a list of contents.
    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        max_size (int): the maximum amount of items in the backpack.
        contents (list): the contents of the backpack.
    Functions:
        put(): Add an item to the backpack.
        take(): Remove and item from the backpack.
        dump(): Remove all the contents from the backpack.
    """

    def __init__(self, name, color, max_size=5):  # This function is the constructor.
        """Set the name, color, and max_size and initialize an empty list of contents.
        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the maximum amount of items in the backpack.
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Check if there is room in the backpack. If there is,
        add 'item' to the backpack's list of contents. Else, tell the user that there is no room."""
        if len(self.contents) >= self.max_size:
            print("No Room!")
            return
        else:
            self.contents.append(item)  # Use 'self.contents', not just 'contents'.

    def take(self, item):
        """Remove 'item' from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Dump all of the contents from the backpack."""
        self.contents = []

    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """If 'self' has fewer contents than 'other', return True.
        Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """If 'self' is the same as 'other', return True.
        Otherwise, return False.
        """
        return self.name == other.name and self.color == other.color and len(self.contents) == \
               len(other.contents)

    def __str__(self):
        """Construct a return a string representation for this backpack."""
        return "Owner:\t\t{} \nColor:\t\t{} \nSize:\t\t{} \nMax Size:\t{} \
            \nContents:\t{}".format(self.name, self.color, len(self.contents), self.max_size, self.contents)

pack1 = Backpack("Rose", "red")
pack2 = Backpack("Rose", "red")
print(pack1)
pack1==pack2

#4
class ComplexNumber:

    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def conjugate(self):
        conj = ComplexNumber(self.real, -self.imag)
        return conj

    def __str__(self):
        if self.imag >= 0:
            s = "({}+{}j)".format(self.real, self.imag)
        else:
            s = "({}{}j)".format(self.real, self.imag)
        return s

    def __abs__(self):
        return (self.real ** 2 + self.imag ** 2) ** .5

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        new_real = self.real + other.real
        new_imag = self.imag + other.imag
        return ComplexNumber(new_real, new_imag)

    def __sub__(self, other):
        new_real = self.real - other.real
        new_imag = self.imag - other.imag
        return ComplexNumber(new_real, new_imag)

    def __mul__(self, other):
        new_real = self.real * other.real - self.imag * other.imag
        new_imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(new_real, new_imag)

    def __truediv__(self, other):
        t = self * other.conjugate()
        b = (other * other.conjugate()).real
        return ComplexNumber(t.real / b, t.imag / b)


def test_ComplexNumber(a, b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")
    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)
    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
    if abs(py_cnum) != abs(my_cnum):
        print("__abs__() failed for", py_cnum)
    if my_cnum != my_cnum:
        print("__eq__() failed for", py_cnum)
    if py_cnum + py_cnum != my_cnum + my_cnum:
        print("__add__() failed for", py_cnum)
    if py_cnum - py_cnum != my_cnum - my_cnum:
        print("__sub__() failed for", py_cnum)
    if py_cnum * py_cnum != my_cnum * my_cnum:
        print("__mul__() failed for", py_cnum)
    if py_cnum / py_cnum != my_cnum / my_cnum:
        print("__truediv__() failed for", py_cnum)

test_ComplexNumber(3, -4)
test_ComplexNumber(2, 5)

# Exceptions and File/IO

#1
def arithmagic():
    """
    Prompts user for related sequences of 3-digit integers, eventually arriving at 1089.
    Tests each time to see if user enters valid entries. If not, we raise a ValueError.
    """
    step_1 = input("Enter a 3-digit number where the first "
                   + "and last digits differ by 2 or more: ")
    if any(list(step_1)) not in list(range(0, 10)) or len(str(step_1)) != 3:
        raise ValueError("Must be a 3-digit integer.")
    if abs(int(str(step_1)[0]) - int(str(step_1)[-1])) < 2:
        raise ValueError("First and last digits must differ "
                         + "by at least 2.")

    step_2 = input("Enter the reverse of the first number, "
                   + "obtained by reading it backwards: ")
    if str(step_2)[::-1] != str(step_1):
        raise ValueError("Must be the reverse of the first number.")

    step_3 = input("Enter the positive difference of these "
                   + "numbers: ")
    if int(step_3) != abs(int(step_1) - int(step_2)):
        raise ValueError("Must be the positive difference of "
                         + "the first two numbers.")

    step_4 = input("Enter the reverse of the previous result: ")
    if str(step_4)[::-1] != str(step_3):
        raise ValueError("Must be the reverse of the third number.")

    print((str(step_3) + "+" + str(step_4) + "= 1089 (ta-da!)"))

#2
def random_walk(max_iters=1e3):
    """
    Returns a random walk (int) for 1e3 max iterations.
    """
    walk=0
    directions=[1,-1]
    for i in range(int(max_iters)):
        walk+=choice(directions)
    return walk

def random_walk(max_iters=1e10):
    """
    Returns a random walk (int) for 1e3 max iterations.
    Aborts and reports the iteration and walk amounts if ^C is entered by user before completion of calculation.
    """
    iteration = 0
    try:
        walk=0
        directions=[1,-1]
        for i in range(int(max_iters)):
            walk+=choice(directions)
            iteration += 1
    except KeyboardInterrupt:
        print("Process interrupted at iteration " + str(iteration))
    else:
        print("Process completed")
    finally:
        return walk

#3
class ContentFilter:

    def __init__(self, file=''):
        try:
            with open(file, 'r') as f:
                print("Valid to open")
                self.file = file
                self.content = f.readlines()
                # with automatically will f.close()
        except:
            file1 = input('Please enter a valid file name ')
            self.__init__(file1)

test1 = ContentFilter('test.txt')

#4
class ContentFilter:

    def __init__(self, file=''):
        try:
            with open(file, 'r') as f:
                self.file = file
                self.content = f.readlines()
                print("Valid to open")
                # with automatically will f.close()

                self.char = sum([len(line) for line in self.content])
                self.alpha = sum([word.isalpha() for line in self.content for word in line])
                self.numer = sum([word.isdigit() for line in self.content for word in line])
                self.white = sum([word.isspace() for line in self.content for word in line])
                self.numLine = len(self.content)


        except (FileNotFoundError, TypeError, OSError) as e:
            print(e)
            file1 = input('Please enter a valid file name ')
            self.__init__(file1)

    def uniform(self, outFile='', mode='w', case='upper'):
        try:
            if mode not in ['w', 'x', 'a']:
                raise ValueError("Mode must be 'w', 'r', or 'a '")
            elif case.strip() not in ['lower', 'upper']:
                raise ValueError("Case must be either upper or lower")

            with open(outFile, mode) as outFile:
                if case.strip() == 'upper':
                    for line in self.content:
                        outFile.write(line.upper())
                else:
                    for line in self.content:
                        outFile.write(line.lower())
        except ValueError as e:
            print('VALUE ERROR: ', e)

    def reverse(self, outFile, mode='w', unit='line'):
        try:
            if unit.strip() not in ['word', 'line']:
                raise ValueError('Unit must be either line or word')

            with open(outFile, mode) as outFile:
                if unit.strip() == 'line':
                    for line in list(reversed(self.content)):
                        outFile.write(line + '\n')
                else:
                    for line in self.content:
                        outFile.write(line[::-1] + '\n')
        except ValueError as e:
            print('ValueError: ', e)

    def transpose(self, outFile='', mode='w', case=''):
        numWords = 3
        with open(outFile, mode) as outFile:
            for i in range(numWords + 2):
                if i != 0:
                    outFile.write('\n')
                for j, line in enumerate(self.content):
                    to_add = line[i].strip() + ' '
                    outFile.write(to_add)

    def __str__(self):
        text = "Source file:\t\t <{}>\n".format(self.file)
        text += "Total characters:\t < {} >\n".format(self.char)
        text += "Alphabetic characters:\t < {} >\n".format(self.alpha)
        text += "Numerical characters:\t < {} >\n".format(self.numer)
        text += "Whitespace characters:\t < {} >\n".format(self.white)
        text += "Number of lines:\t < {} >\n".format(self.numLine)

        return text

test = ContentFilter('test.txt')
test.content
print(test)