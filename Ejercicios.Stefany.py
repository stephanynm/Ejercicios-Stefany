#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
math.sqrt(25)
math.pi
import math as mt
mt.sqrt(25)


# In[2]:


from numpy import random
import numpy as np
# uniform random numbers in [0,1]
dataOne = random.rand(5,5)


# In[3]:


dataOne


# In[4]:


np.mean(dataOne)


# In[5]:


print('I Love data Science')


# In[17]:


varOne = 25
varTwo = 25.0
varThree = varOne + varTwo
print (varThree)


# In[19]:


varOne = 25
varTwo = 'Hello'
varThree = varOne + varTwo
print(varThree)


# In[9]:


listOne = [1,2,3,4]
print(listOne[1:3])


# In[10]:


tel  =  { 'jack' :  4098 ,  'sape' :  4139 }
tel ['jack']


# In[11]:


precio  =  { 'home1' :  100000 ,  'home2' :  90000, 'home3' :  80000 , 'home4' :  80000 , 'home5' :  70000 }
precio ['home2']


# In[14]:


var1= precio ['home1']
var1


# In[20]:


var2= precio ['home2']
var3= precio ['home3']
var4= var2+var3
print(var4)


# In[21]:


var5= var1-var2
print (var5)


# In[22]:


var6= var4+var2
print (var6)


# In[24]:


the_world_is_flat = True
if the_world_is_flat:
    print("Be careful not to fall off!")


# In[25]:


# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."


# In[26]:


2 + 2
4
50 - 5*6
20
(50 - 5*6) / 4
5.0
8 / 5  # division always returns a floating point number
1.6


# In[28]:


17 / 3  # classic division returns a float


# In[29]:


17 // 3  # floor division discards the fractional part


# In[30]:


17 % 3  # the % operator returns the remainder of the division


# In[31]:


5 * 3 + 2  # result * divisor + remainder


# In[35]:


5 ** 2  # 5 squared


# In[36]:


2 ** 7  # 2 to the power of 7


# In[37]:


width = 20
height = 5 * 9
width * height


# In[38]:


n  # try to access an undefined variable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'n' is not defined


# In[39]:


4 * 3.75 - 1


# In[40]:


tax = 12.5 / 100
price = 100.50
price * tax


# In[41]:


price + _


# In[42]:


round(_, 2)


# In[43]:


'spam eggs'  # single quotes


# In[44]:


'doesn\'t'  # use \' to escape the single quote...


# In[45]:


"doesn't"  # ...or use double quotes instead


# In[46]:


'"Yes," they said.'


# In[47]:


"\"Yes,\" they said."


# In[48]:


'"Isn\'t," they said.'


# In[49]:


'"Isn\'t," they said.'


# In[50]:


print('"Isn\'t," they said.')


# In[51]:


s = 'First line.\nSecond line.'  # \n means newline
s  # without print(), \n is included in the output


# In[52]:


print(s)  # with print(), \n produces a new line


# In[53]:


print('C:\some\name')  # here \n means newline!


# In[54]:


print(r'C:\some\name')  # note the r before the quote


# In[55]:


print("""Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")


# In[56]:


# 3 times 'un', followed by 'ium'
3 * 'un' + 'ium'


# In[57]:


'Py' 'thon'


# In[58]:


text = ('Put several strings within parentheses '
        'to have them joined together.')
text


# In[59]:


prefix = 'Py'
prefix 'thon'  # can't concatenate a variable and a string literal
  File "<stdin>", line 1
    prefix 'thon'
                ^


# In[64]:


prefix = 'Py'
prefix 'thon'  # can't concatenate a variable and a string literal
  File "<stdin>", line 1
    prefix 'thon'
                ^


# In[65]:


prefix + 'thon'


# In[66]:


word = 'Python'
word[0]  # character in position 0


# In[67]:


word[5]  # character in position 5


# In[68]:


word[-1]  # last character


# In[69]:


word[-2]  # second-last character


# In[70]:


word[-6]


# In[71]:


word[0:2]  # characters from position 0 (included) to 2 (excluded)


# In[72]:


word[2:5]  # characters from position 2 (included) to 5 (excluded)


# In[73]:


word[:2] + word[2:]


# In[74]:


word[:4] + word[4:]


# In[75]:


word[:2]


# In[76]:


word[42]  # the word only has 6 characters
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>


# In[77]:


word[4:42]


# In[78]:


word[42:]


# In[79]:


word[0] = 'J'


# In[80]:


word[2:] = 'py'


# In[81]:


word[2:] = 'py'


# In[82]:


word[:2] + 'py'


# In[83]:


s = 'supercalifragilisticexpialidocious'
len(s)


# In[84]:


squares = [1, 4, 9, 16, 25]
squares


# In[85]:


squares[0]  # indexing returns the item


# In[86]:


squares[-1]


# In[87]:


squares[-3:]  # slicing returns a new list


# In[88]:


squares[:]


# In[89]:


squares + [36, 49, 64, 81, 100]


# In[91]:


cubes = [1, 8, 27, 65, 125]  # something's wrong here
4 ** 3  # the cube of 4 is 64, not 65!


# In[92]:


cubes[3] = 64  # replace the wrong value
cubes


# In[93]:


cubes.append(216)  # add the cube of 6
cubes.append(7 ** 3)  # and the cube of 7
cubes


# In[94]:


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters


# In[95]:


# replace some values
letters[2:5] = ['C', 'D', 'E']
letters


# In[96]:


# now remove them
letters[2:5] = []
letters


# In[97]:


letters[:] = []
letters
[]


# In[98]:


letters = ['a', 'b', 'c', 'd']
len(letters)
4


# In[99]:


a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
x


# In[100]:


x[0]


# In[101]:


x[0][1]


# In[102]:


# Fibonacci series:
# the sum of two elements defines the next
a, b = 0, 1
while a < 10:
    print(a)
    a, b = b, a+b


# In[103]:


i = 256*256
print('The value of i is', i)


# In[104]:


a, b = 0, 1
while a < 1000:
    print(a, end=',')
    a, b = b, a+b


# In[1]:


x = int(input("Please enter an integer: "))


# In[2]:


if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')


# In[4]:


... words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))


# In[7]:


for i in range(5):
    print(i)


# In[8]:


range(5, 10)


# In[9]:


range(0, 10, 3)


# In[10]:


a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
    print(i, a[i])


# In[11]:


print(range(10))


# In[12]:


sum(range(4))  # 0 + 1 + 2 + 3


# In[13]:


list(range(4))


# In[14]:


for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')


# In[15]:


for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number", num)
        continue
    print("Found a number", num)


# In[1]:


import numpy
import matplotlib
import pandas
import scipy
import -U scikit-learn


# In[2]:


def fib(n):    # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()


# In[3]:


# Now call the function we just defined:
fib(2000)


# In[5]:


fib
f = fib
f(100)


# In[6]:


fib(0)
print(fib(0))


# In[7]:


def fib2(n):  # return Fibonacci series up to n
    """Return a list containing the Fibonacci series up to n."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)    # see below
        a, b = b, a+b
    return result

f100 = fib2(100)    # call it
f100                # write the result


# In[8]:


def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


# In[9]:


i = 5

def f(arg=i):
    print(arg)

i = 6
f()


# In[10]:


def f(a, L=[]):
    L.append(a)
    return L

print(f(1))
print(f(2))
print(f(3))


# In[11]:


def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")


# In[12]:


parrot(1000)                                          # 1 positional argument
parrot(voltage=1000)                                  # 1 keyword argument
parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword


# In[14]:


def function(a):
    pass

function(0, a=0)


# In[15]:


def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])


# In[16]:


cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")


# In[20]:


standard_arg(2)


# In[21]:


standard_arg(arg=2)


# In[33]:


def foo(name, /, **kwds):
foo(1, **{'name': 2})


# In[34]:


def write_multiple_items(file, separator, *args):
    file.write(separator.join(args))


# In[35]:


def concat(*args, sep="/"):
    return sep.join(args)

concat("earth", "mars", "venus")


# In[36]:


concat("earth", "mars", "venus", sep=".")


# In[37]:


list(range(3, 6))            # normal call with separate arguments


# In[38]:


args = [3, 6]
list(range(*args))            # call with arguments unpacked from a list


# In[39]:


def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")


# In[40]:


d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)


# In[41]:


def make_incrementor(n):
    return lambda x: x + n

f = make_incrementor(42)
f(0)


# In[42]:


f(1)


# In[43]:


pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
pairs


# In[44]:


def my_function():
    """Do nothing, but document it.

    No, really, it doesn't do anything.
    """
    pass

print(my_function.__doc__)


# In[45]:


def f(ham: str, eggs: str = 'eggs') -> str:
    print("Annotations:", f.__annotations__)
    print("Arguments:", ham, eggs)
    return ham + ' and ' + eggs

f('spam')


# In[1]:


fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
fruits.count('apple')


# In[2]:


fruits.count('tangerine')


# In[3]:


fruits.index('banana')


# In[4]:


fruits.index('banana', 4)  # Find next banana starting a position 4


# In[5]:


fruits.reverse()
fruits


# In[6]:


fruits.append('grape')
fruits


# In[7]:


fruits.sort()
fruits


# In[8]:


fruits.pop()


# In[9]:


stack = [3, 4, 5]
stack.append(6)
stack.append(7)
stack


# In[10]:


stack.pop()


# In[11]:


stack


# In[12]:


stack.pop()


# In[13]:


stack.pop()


# In[14]:


stack


# In[15]:


from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")           # Terry arrives
queue.append("Graham")          # Graham arrives
queue.popleft()                 # The first to arrive now leaves


# In[16]:


queue.popleft()                 # The second to arrive now leaves


# In[17]:


queue                           # Remaining queue in order of arrival


# In[18]:


squares = []
for x in range(10):
    squares.append(x**2)

squares


# In[19]:


squares = list(map(lambda x: x**2, range(10)))


# In[20]:


squares = [x**2 for x in range(10)]


# In[21]:


[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]


# In[22]:


combs = []
for x in [1,2,3]:
    for y in [3,1,4]:
        if x != y:
            combs.append((x, y))

combs


# In[23]:


vec = [-4, -2, 0, 2, 4]
# create a new list with the values doubled
[x*2 for x in vec]


# In[24]:


# filter the list to exclude negative numbers
[x for x in vec if x >= 0]


# In[25]:


# apply a function to all the elements
[abs(x) for x in vec]


# In[26]:


# call a method on each element
freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
[weapon.strip() for weapon in freshfruit]


# In[27]:


# create a list of 2-tuples like (number, square)
[(x, x**2) for x in range(6)]


# In[28]:


# the tuple must be parenthesized, otherwise an error is raised
[x, x**2 for x in range(6)]


# In[29]:


# flatten a list using a listcomp with two 'for'
vec = [[1,2,3], [4,5,6], [7,8,9]]
[num for elem in vec for num in elem]


# In[30]:


from math import pi
[str(round(pi, i)) for i in range(1, 6)]


# In[31]:


matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]


# In[32]:


[[row[i] for row in matrix] for i in range(4)]


# In[33]:


transposed = []
for i in range(4):
    transposed.append([row[i] for row in matrix])

transposed


# In[34]:


transposed = []
for i in range(4):
    # the following 3 lines implement the nested listcomp
    transposed_row = []
    for row in matrix:
        transposed_row.append(row[i])
    transposed.append(transposed_row)

transposed


# In[35]:


list(zip(*matrix))


# In[36]:


a = [-1, 1, 66.25, 333, 333, 1234.5]
del a[0]
a


# In[37]:


del a[2:4]
a


# In[38]:


del a[:]
a


# In[39]:


del a


# In[40]:


t = 12345, 54321, 'hello!'
t[0]


# In[41]:


t


# In[42]:


# Tuples may be nested:
u = t, (1, 2, 3, 4, 5)
u


# In[43]:


# Tuples are immutable:
t[0] = 88888


# In[44]:


... v = ([1, 2, 3], [3, 2, 1])
v


# In[45]:


empty = ()
singleton = 'hello',    # <-- note trailing comma
len(empty)


# In[46]:


len(singleton)


# In[47]:


singleton


# In[48]:


x, y, z = t


# In[49]:


basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)                      # show that duplicates have been removed


# In[50]:


'orange' in basket                 # fast membership testing


# In[51]:


'crabgrass' in basket


# In[52]:


# Demonstrate set operations on unique letters from two words

a = set('abracadabra')
b = set('alacazam')
a                                  # unique letters in a


# In[53]:


a - b                              # letters in a but not in b


# In[54]:


a | b   # letters in a or b or both


# In[55]:


a & b                              # letters in both a and b


# In[56]:


a ^ b                              # letters in a or b but not both


# In[57]:


a = {x for x in 'abracadabra' if x not in 'abc'}
a


# In[58]:


tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
tel


# In[59]:


tel['jack']


# In[60]:


del tel['sape']
tel['irv'] = 4127
tel


# In[61]:


list(tel)


# In[62]:


sorted(tel)


# In[63]:


'guido' in tel


# In[64]:


'jack' not in tel


# In[65]:


dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])


# In[66]:


{x: x**2 for x in (2, 4, 6)}


# In[67]:


dict(sape=4139, guido=4127, jack=4098)


# In[68]:


knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)


# In[69]:


for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)


# In[70]:


questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q, a))


# In[71]:


for i in reversed(range(1, 10, 2)):
    print(i)


# In[72]:


basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)


# In[73]:


import math
raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
filtered_data = []
for value in raw_data:
    if not math.isnan(value):
        filtered_data.append(value)

filtered_data


# In[74]:


string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
non_null = string1 or string2 or string3
non_null


# In[75]:


(1, 2, 3)              < (1, 2, 4)
[1, 2, 3]              < [1, 2, 4]


# In[76]:


(1, 2, 3, 4)           < (1, 2, 4)
(1, 2)                 < (1, 2, -1)
(1, 2, 3)             == (1.0, 2.0, 3.0)
(1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)


# In[83]:


import csv
 
with open(r'C:\Users\Eric\Documents\Data Analytics\Parte 2\3. Regresión múltiple\newproductattributes2017.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        print(row)


# In[ ]:




