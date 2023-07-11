'''
Activation function apply a non linear  transformation and decide whether a neauron  should be activated or not.
Model will not perform if its only linear,  non linear transformation on the network can learn better and 
perform more  complex task.
After each layer we apply activation func.
Popular activation func - step func(0 to 1), sigmoid(0 to 1),  tanh (-1 to 1), Relu(0((-ve value) to +value),
leakyRelu( a.x((-ve value) to +value if x>0) improved vanishing gradient, softmax
'''
import torch
import torch.nn as nn
import torch.functional as F

# option 1 create nn modules

class NeuralNet (nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # can use any activation function
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
# Option 2 - Apply activation function directly in forward pass

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
    def forward(self, x):
        #F.leaky_relu()
        out = torch.relu(self.linear1(x))    
        out = torch.sigmoid (self.linear2(out))
        return out




        

    
