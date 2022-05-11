# Chain rule
# x ----> a(x) ----> y ----> a(y) ----> z
# dz/dx = dz/dy . dz/dx

# Computational Graph
# x & y ----> f(x.y) ----> z

# calculating local gradient
# dz/dx = d x.y/dx = y
# dz/dy = d x.y/dy = x

# so we want to find the dLoss/dx = dLoss/dz . dz/dx

# first we do forward pass then compute local grad then backward propagation

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute loss
y_hat = w * x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)