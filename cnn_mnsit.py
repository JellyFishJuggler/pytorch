import torch
import torch.nn as nn
import torch.optim as op
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# import time

# start = time.time()

print(torch.cuda.is_available())
device = torch.device("cuda")
print(torch.cuda.get_device_name(0))

train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

testing_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)
# print(len(train_data)) # print(train_data[0]) # print(train_data[0][0].shape)

train_load = DataLoader(train_data, batch_size=64, shuffle=True)
test_load = DataLoader(testing_data, batch_size=64, shuffle=False)

X_batch, y_batch = next(iter(train_load))
# # print(X_batch.shape) # print(y_batch.shape)


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,16,3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(400,10)
    
    def forward(self, x):
        x = self.conv1(x)
        # print("conv:", x.shape)
        x = self.relu1(x)
        # print("relu:", x.shape)
        x = self.pool1(x)
        
        x = self.conv2(x)
        # print("conv:", x.shape)
        x = self.relu2(x)
        # print("relu:", x.shape)
        x = self.pool2(x)
        # print("pool:", x.shape)
        x = torch.flatten(x, 1)
        # print("flatten:", x.shape)
        return self.fc(x)


model = CNN()
model.to(device)
# print(model(X_batch))
# print(torch.argmax(model(X_batch),dim=1))
loss_fn = nn.CrossEntropyLoss()
optimizer = op.SGD(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    for x_i, y_i in train_load:
        
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        
        optimizer.zero_grad()
        y_i_predict = model(x_i)
        loss = loss_fn(y_i_predict, y_i)
        loss.backward()
        optimizer.step()
    # print("Training time:", time.time() - start)
        
        
model.eval()

correct = 0 
total = 0
test_loss = 0
with torch.no_grad():
    
    for x_i,y_i in test_load:
        
        x_i = x_i.to(device)
        y_i = y_i.to(device)
 
        logits = model(x_i)
        ypred = torch.argmax(logits, dim=1)
        loss = loss_fn(logits,y_i)
        
        correct += (y_i == ypred).float().sum()
        total += len(y_i)
        
        test_loss += loss.item()
        # print(f'Test Loss {loss}')
        # print(f'Predictions: {ypred} | logits: {logits}')

test_loss /= len(test_load)
accuracy = correct / total

print("Test Loss:", test_loss)
print("Accuracy:", accuracy.item())