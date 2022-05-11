# Kind of creating multiple generator function then one tight function
# So, incase if you need one function for any other task u can use it

# # Tight function example
# def get_odd_num_sqr_and_ending_in_1():
#     for n in range(1000):
#         if n % 2 != 0: # Odd
#             n **= 2 # Squared
#             if n % 10 ==1: # Ending in 1
#                 print('Match Found --> {}'.format(n))

# get_odd_num_sqr_and_ending_in_1()

# Now with generator function
def odd_filter(nums):
    for num in nums:
        if num % 2 == 1:
            yield num

def squared(nums):
    for num in nums:
        yield num ** 2

def ending_in_1(nums):
    for num in nums:
        if num % 10 == 1:
            yield num

def convert_to_string(nums):
    for num in nums:
        yield 'Match Found -->' +str(num)

get_pipeline = convert_to_string(ending_in_1(squared(odd_filter(range(1000)))))

for n in get_pipeline:
    print(n)
