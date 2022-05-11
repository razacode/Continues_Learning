from functools import wraps
from time import sleep, perf_counter
import random

from torch import rand

def get_secure_message(func):
    "This is hello message function"
    @wraps(func)
    def wrapper(n1, n2):
        for n in range(5):
            t_start = perf_counter()
            func(n1, n2)
            t_end = perf_counter()
            duration = t_end - t_start
            print(round(duration, 3))
        return 'S3cR3T'
    return wrapper

@get_secure_message
def secure_message(n,m):

    sleep(random.randint(n,m))
    return 'hello'
print(secure_message(1,6))