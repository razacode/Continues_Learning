# 1) Desgin model and give input, output feature and forward pass
# 2) construct loss and optimizer
# 3) Traning loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare dataset
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

#  cast to float tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
# print(X)
y = torch.from_numpy(y_numpy.astype(np.float32))
# print(y)
y = y.view(y.shape[0], 1)
# print(y)

n_samples, n_features = X.shape

# 1)  Model 
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2)  loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Traning loop
n_epoch = 100
for epoch in range(n_epoch):
    # Forward pass and loop
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad 
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.3f}')

# plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()