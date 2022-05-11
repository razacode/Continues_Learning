# gen_send allow send something to generator

# def simple_gen(number=10):
#     for n in range(number):
#         val = yield n * 2
#         print("received ", val)

# gen1 = simple_gen()
# print(gen1.__next__())
# # here we will use send
# print(gen1.send(78))
# print(gen1.__next__())
# print(gen1.send(17))
# print(gen1.__next__())
# print(gen1.__next__())

# # Complex example

# def simple_gen(number=10):
#     for n in range(number):
#         val = yield number * 2
#         print("received ", val)
#         if val:
#             number += val
#         else:
#             number += 1

# gen1 = simple_gen()
# print(gen1.__next__())
# # here we will use send
# print(gen1.send(30))
# print(gen1.__next__())
# print(gen1.send(50))
# print(gen1.__next__())
# print(gen1.__next__())

# While loop
from distutils.util import strtobool


def simple_generator(start_number=10):
    i = start_number
    while True:
        x = (yield i * 2)
        if x:
            i += x
        else:
            i +=1

gen1 = simple_generator()
print(gen1.__next__())
# here we will use send
print(gen1.send(30))
print(gen1.__next__())
print(gen1.send(20))
print(gen1.__next__())
print(gen1.__next__())
print(gen1.send(90))
print(gen1.__next__())
print(gen1.send(80))
print(gen1.__next__())
print(gen1.__next__())

