animals = ['dog', 'cat', 'hen', 'fox']

# TODO Generator Expression

animals_upper = (animal.upper() for animal in animals)

# TODO print
print(list(animals_upper))

#####################################