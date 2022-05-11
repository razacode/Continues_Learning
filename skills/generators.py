# Generator functions allow you to declare a function that behaves like an iterator, 
# i.e. it can be used in a for loop.

# Normal function

# def function():
#     return 100

# x = function()
# print(x)

# With generator we will replace with yeild and __next__

from re import X
from pkg_resources import yield_lines

# def my_generator():
#     yield 100

# x = my_generator()
# print(x.__next__())

# # Now if we have more then one yeild

# def my_generator():
#     yield 100
#     yield 140
#     yield 130

# x = my_generator()
# print(x.__next__())
# print(x.__next__())
# print(x.__next__())

# Now if we have more then one yeild instead of passing one by one next we will pass FOR loop

def my_generator():
    yield 100
    yield 140
    yield 130

x = my_generator()
for n in x:
    print(n)