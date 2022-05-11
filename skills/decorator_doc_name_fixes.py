from functools import wraps

# from skills.decorator_doc_name_fixes import hello

def hello_message_new(func):

    "This is hello message decorator"
    @wraps(func)
    def wrapper():
        "This is wrapper inside message."
        return ">>>> Wrapper Hello <<<<"
    return wrapper

@hello_message_new
def hello_message():
    "This is hello message"
    return 'hello'

print(hello_message())