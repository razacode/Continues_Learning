# def my_sqr_fun_gen(n):
#     for i in range(n):
#         yield i ** 2
    
# sqr_num = my_sqr_fun_gen(6)
# for n in sqr_num:
#     print(n)


# For the above generator function can be replace with expression
    
sqr_num = (n ** 2 for n in range(6))
# print(list(sqr_num))

# OR
for num in sqr_num:
    print(num)