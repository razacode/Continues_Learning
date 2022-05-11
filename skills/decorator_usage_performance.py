# Here we will check our function performance

from functools import wraps
from time import sleep, perf_counter
import random

def hello_message_new(func):

    "This is hello message decorator"
    @wraps(func)
    def wrapper():
        for n in range(5):
            t_start = perf_counter()
            func()
            t_end = perf_counter()
            duration = t_end - t_start
            print(round(duration, 3))
    return wrapper

@hello_message_new
def hello_message():
    "This is hello message"
    sleep(random.randint(1,5))
    return 'hello'

print(hello_message())