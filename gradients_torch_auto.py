# this module is for prediction auto
#1. Design model(input, output size, forward pass)
#2. Construct loss and optimizer
#3. Training Loop
#   - forward pass : compute the prediction
#   - Backward pass :gradients
#   - update weights

import torch
import torch.nn as nn

# f = w*x
#f =2*x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

#w= torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#in place of weight w, X, Y in 2D and below changes:

n_samples, n_features = X.shape
print(n_samples,n_features)

input_size = n_features
output_size = n_features

#model prediction
#model = nn.Linear(input_size,output_size)

# creating class inplace of model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training : f(5) = {model(X_test).item():.3f}')

#Training
learning_rate=0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward_pass
    l.backward() #dl/dw

    #update weight
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()
            
    if epoch % 10 ==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss={l:.8f}')

print(f'Prediction after training : f(5)={model(X_test).item():.3f}')
