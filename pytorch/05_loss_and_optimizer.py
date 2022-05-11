# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f"samples {n_samples}, featuers {n_features}")

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design Model, the model has implement forward pass
input_size = n_features
output_size = n_features

# we call this model
model = nn.Linear(input_size, output_size)

# # Or use this class we can define
# class LinearRegresion(nn.Module):
#     def __init__(self, input_dim,output_dim):
#         super(LinearRegresion, self).__init__()
#         self.lin = nn.Linear(input_dim, output_dim)

#     def foward(self, x):
#         return self.lin(x)

# model = LinearRegresion(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # predict forward pass
    y_predited = model(X)

    # loss
    l = loss(Y, y_predited)

    # calculate gradients = backward
    l.backward()

    # update weight
    optimizer.step()

    # zero gradient after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1} : w = { w[0][0]:.3f} loss = {l:.3f}')

print(f'Prediction after training f(5) = {model(X_test).item():3f}')