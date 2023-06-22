import torch

#x= torch.rand(2,2)
#y= torch.rand(2,2)
#print(x)
#print(y)
#z=print(x+y)
#print(torch.add(x,y))

#print(torch.zeros(3,3))
#print(torch.ones(2,2))


#print(x.dtype)
#a= torch.ones(2,2,dtype=torch.double)# can try double, int, float
#print("Data type is", a.dtype)
#print("tensor size is", a.size())

#print(torch.mul(x,y))
#print(torch.sub(x,y))
#print(torch.divide(x,y))

c = torch.rand(4,4)
print(c)
print(c.view(16)) #16 is max from 4*4
d=c.view(-1,4)
print(d)
print(d.size())

#print(x.add_(y)) #x+y using add_

#print(x[1,1])

# torch to numpy and viceversa

import numpy as np
x= torch.ones(5)
print(x)
y=x.numpy()
print(y)
print(type(y))
# if we are using CPU botn numpy and torch share same memory, so any change will change in both.


#numpy to torch
x1= np.ones(5)
print(x1)
y1= torch.from_numpy(x1) # numpy to torch
print(y1)
x1+=1
print(x1)
print(y1)
