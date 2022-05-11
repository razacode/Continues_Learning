# Here we will look for the decorator concept
# how the control of function is pass to other function

###################################
# Decorators are a very powerful and useful tool in Python since it allows programmers 
# to modify the behaviour of function or class. Decorators allow us to wrap another function 
# in order to extend the behaviour of the wrapped function, without permanently modifying it.

def hello_message_decorator(func):

    def wrapper():
        return '<<<< Wrapper Hello <<<<'
    return wrapper
# here we will call decorator
@hello_message_decorator
def hello():
    return 'hello'

print(hello())

