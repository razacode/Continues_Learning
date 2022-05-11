# def my_number_square(n):
#     result = []
#     for x in range(n):
#         result.append(x **2)
#     return result

# a = my_number_square(5)
# print(a)

# Now with generator

def my_number_square_gen(n):
    for x in range(n):
        yield x * x

# # with list method
# print(list(my_number_square_gen(5)))

# # with __next__ 
# gen = my_number_square_gen(5)
# print(gen.__next__())
# print(next(gen))
# print(next(gen))

# with for loo[]
gen = my_number_square_gen(5)
for n in gen:
    print(n)