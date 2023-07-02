'''
epoch = 1 forward and 1 backward pass of all training samples
batch_size = number of training samples in one forward and backward pass
number of iteration = number of passes, each pass using (batch_size)no. of samples
ex: 100 samples, batch_size=20 -->100/20 = 5 iteration for 1 epoch
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('.\pytorch_learning\data\Dataset\winedata.txt', delimiter=",", skiprows=1,dtype=np.float32)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()

#first_data = dataset[:2]
#features, labels =first_data
#print(features,labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

#dataiter = iter(dataloader)
#data = dataiter.next()
#features, labels = data
#print(features,labels)


#training loop
num_epoch = 3
total_sample = len(dataset)
n_iteration = math.ceil(total_sample/4)
print(total_sample, n_iteration)

for epoch in range(num_epoch):
    for i, (inputs,labels) in enumerate(dataloader):
        #forwards, backward update
        if (i+1)%5 == 0:
            print(f'epoch:{epoch+1}/{num_epoch}, step:{i+1}/{n_iteration}, inputs:{inputs.shape}')

