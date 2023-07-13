#MNIST data set
# DataLoader, Data Transformation
# Multi layer neural network, activation function
#Loss  and optimizer
#Training Loop(batch training)
#Model Evaluation
#GPU support

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
input_size = 784 # 28x28
hidden_size = 100
batch_size = 100
num_epoch = 2
num_classes = 10 # 0 to 9
learning_rate = 0.001

#MNIST

train_dataset =torchvision.datasets.MNIST(root='.\pytorch_learning\data\Dataset', train=True,
                                          transform=transforms.ToTensor(), download=True)
test_dataset =torchvision.datasets.MNIST(root='.\pytorch_learning\data\Dataset', train=False,
                                          transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size,
                                           shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size,
                                          shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)


for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(samples[i][0], cmap='viridis')
plt.show()


class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss() # we cannot run sigmoid in forward  as we will use CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop

n_total_steps =  len(train_loader)
for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):
        #100,1,28,28
        #100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        #forward pass

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epoch}, step{i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing loop

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        print('labels', labels)
        outputs = model(images)

        #value, item
        _,predictions = torch.max(outputs,1)
        print('predictions',predictions)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    acc = 100.0 *n_correct/n_samples

    print(f'Accuracy : {acc}')

