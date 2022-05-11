from pkg_resources import yield_lines
from sklearn import pipeline

def odd_filter(nums):
    for num in nums:
        if num % 2 != 0:
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
        yield 'Match Found -->' + str(num)

pipeline = ending_in_1(squared(odd_filter(range(1000))))

for n in pipeline:
    if len(str(n)) >= 5:
        # break
        pipeline.throw(ValueError("Too Long!"))
    print(n)

# print(pipeline.__next__())
