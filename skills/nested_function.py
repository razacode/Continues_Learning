# Nested function
def outer_function():

    # Define inner function
    def inner_function():
        print("This is from inner function")

    print("This is from outer function")
    # return inner_function
    inner_function()

outer_function()
# if you want to use return type uncomment below lines
# x = outer_function()
# x()