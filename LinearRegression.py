#1. Design model(input, output size, forward pass)
#2. Construct loss and optimizer
#3. Training Loop
#   - forward pass : compute the prediction
#   - Backward pass :gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare your data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=1, noise=20)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape

# model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop

num_epoch = 100;
for epoch in range(num_epoch):
    #forward pass and losss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    #backward pass
    loss.backward()

    #update
    Optimizer.step()

    #zero grad
    Optimizer.zero_grad()

    if (epoch+1)% 10 == 0:
        print(f'epoch:{epoch+1}, loss= {loss.item(): .4f}')

# plot
predicted = model(X).detach().numpy()  # detach will remove zero gradient and converted to numpy.
plt.plot(X_numpy,y_numpy, 'ro')
plt.plot(X_numpy, predicted,'b')
plt.show()