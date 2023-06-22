import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

#loss 
y_hat = x*w
loss = (y_hat-y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

#update weights
#next forward and backward

w = torch.tensor(2.0, requires_grad=True)

y_hat = x*w
loss = (y_hat-y)**2

print(loss)

loss.backward()
print(w.grad)