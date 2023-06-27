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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

n_samples, n_features = X.shape

#scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convertion from numpy to tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# model
# f= wx+b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.Linear = nn.Linear(n_input_features,1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.Linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# Loss
learning_rate = 0.01
num_epoch = 100

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# update weights

for epoch in range(num_epoch):
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)
    #backward pass
    loss.backward()

    optimizer.step()
    
    optimizer.zero_grad()

    if(epoch+1)%10 ==0:
        print(f'epoch : {epoch+1}, loss :{loss.item():.4f}')

# Evaluation

with torch. no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy: {acc :.4f}')