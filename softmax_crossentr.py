# softmax is to squeze input to output
# Cross entropy is to find loss, low value : good prediction, high value: bad prediction
# in pytorch cross entropy  no softmax in last layer  and Y has class labels not one hot encoded

import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()
'''

for 1 sample
#Y = torch.tensor([0])
#nsamples x nclasses = 1x3

y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[1.5, 2.0, 0.2]])

'''
#for 3 samples
Y = torch.tensor([2, 0, 1])
#nsamples x nclasses = 3x3

y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
y_pred_bad = torch.tensor([[1.5, 2.0, 0.2], [0.5, 2.0, 0.2], [2.1, 1.0, 0.2]])


l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)
print(Y)
print(l1.item())
print(l2.item())

pred, prediction1 = torch.max(y_pred_good, 1)
predb, prediction2 = torch.max(y_pred_bad, 1)

print(predb)
print(prediction1)
print(prediction2)