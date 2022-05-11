# this is the torch version of 04_grad_nupmy here we willnot use gradient function
# will usr bacward(), tensor instead of numpy

import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before trainig: f(5) = {forward(5):.3f}')

# Training
l_r = 0.01
n_iter = 15

for epoch in range(n_iter):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient
    l.backward()

    # update weights
    with torch.no_grad():
        w -= l_r * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:3f}, loss = {l:3f}')

print(f'Prediction after trainig: f(5) = {forward(5):.3f}')