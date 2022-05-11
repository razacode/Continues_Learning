import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

z = z.mean()
print(z)

# Here we are propagating z back to update value of x
z.backward()
print(x.grad)

# tracking history and not to attach grad_fn in out put in 3 way

a = torch.randn(4, requires_grad=True)
print(a)

# 1st
a.requires_grad_(False)
print(a)

# 2nd
b = a.detach()
print(b)

# 3rd
with torch.no_grad():
    c = a + 3
    print(c)

weights = torch.ones(5, requires_grad=True)

for epoch in range(4):
    model_output = (weights*2).sum()

    model_output.backward()
    print(weights.grad)

    # this below line will refresh grad after everyline
    # weights.grad.zero_()
