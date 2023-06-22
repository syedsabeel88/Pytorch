import torch

x = torch.randn(3,requires_grad=True)
print(x)

y = x+2
print(y)

z=y*y+2
#z=z.mean()
#print(z)

#z.backward() #dz/dx , if its not scalar then we have to provide vector in argument
#print(x.grad)


z1=y*y+2
#print(z1)
v= torch.tensor([0.1,0.001,1.0], dtype=torch.float32)
z.backward(v) # if not scalar, pass vector
#print(x.grad)

#preventing gradient function from below 3 methods:
# x.requires_grad_(False)
#x.detach()
#with torch.no_grad():

x.requires_grad_(False)
print(x)

y =x.detach()
print("detach",y)


with torch.no_grad():
    y=x+2
    print("torch no_grad", y)





# to check weight gradient

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()
    print("weights_grad", weights.grad)

    weights.grad.zero_()  # this will make zero gradient and give same output