# Will see the wrapper concept

from email import message
from requests import ReadTimeout


def hello_message_decorator(func):

    # This is our wripper function
    def wrapper():
        msg = func()
        print("From Decoratot: ", msg)
        return ">>>> Wrapper Hello <<<<"
    return wrapper

def message_hello():
    return 'hello'

message =hello_message_decorator(message_hello)
print(message())