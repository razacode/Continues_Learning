from tkinter import Y
from matplotlib import artist
from numpy import double
import torch
from zmq import device

# x = torch.zeros(2,3)
# x = torch.empty(2,3)
x = torch.rand(2,3)
print(x)

# giving datatype
y = torch.ones(3,3, dtype=torch.int)
print(y.dtype)

# Addition of two tensor
a = torch.rand(2,2)
b = torch.rand(2,2)
print('Printing value of a, b and c')
print(a)
print(b)
# c = a + b
c = torch.add(a,b)
print(c)

# _ in pytorch do inplace operation like in above b.add_(a) b value will be replace with a and b addition

# Slicing operation

print('Slicing operation')
s = torch.rand(5,3)
print(s)
print(s[:,1])

# Resizing operation

print('Resizing operation')
r = torch.rand(3,4)
print(r)
re = r.view(4,3)
print(re.size())
print(re)

# tensor to numpy array
ar_t = torch.ones(6)
print(ar_t)
ar_np = ar_t.numpy()
# ar_np = torch.from_numpy(ar_t)
print(ar_np)
print(type(ar_np))

# Now point is that they share the same memory point to same memoery see below
# happend only when tensor are in GPU 
ar_t.add_(1)
print('Added 1: ', ar_t)
# same print now numpy array
print('without added 1: ', ar_np) # it still add 1 so be careful

# How to execute code on GPU
print('executing code on GPU')
if torch.cuda.is_available():
    device = torch.device('cuda')
    g = torch.ones(5, device=device)
    # or pass to device
    g1 = torch.ones(5)
    g1 = g1.to(device)
    su_m = g + g1
    print(su_m)
    su_m = su_m.to('cpu')
