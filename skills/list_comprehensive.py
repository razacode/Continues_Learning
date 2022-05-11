# num = [1,2,3,4,5]
# # print(num)
# new_num = []

# for n in num:
#     new_num.append(n+n)
# print(new_num)

# # Or you can replace by Comprehensive

# num = [1,2,3,4,5]
# new_num1 = [n + n for n in num]
# print("Using Comprehensive")
# print(new_num1)

from unicodedata import name

print("Using name == __main__")
num = [1,2,3,4,5,6,7,8,9]

def main():
    new_num = []
    for n in num:
        new_num.append(n+n)
    print("new_num: ",new_num)

    # TODO: Using Comprehensive
    new_num1 = [n * n for n in num]
    print("new_num1: ",new_num1)

    # TODO: Using Comprehensive & condition 
    new_num2 = [n + n for n in num if n%2 == 0]
    print("new_num2: ",new_num2)

    # TODO: Using Comprehensive & condition (range)
    new_num3 = [n + n for n in num if 2 < n < 8]
    print("new_num3: ", new_num3)

if __name__ == '__main__':
    main()