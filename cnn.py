import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from torchvision import transforms


# image = Image.open('digit.png')
# transform = transforms.ToTensor()
# x = transform(image)
# print(x.shape)

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,8,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(1352,3)
    
    def forward(self, x):
        x = self.conv(x)
        # print("conv:", x.shape)
        x = self.relu(x)
        # print("relu:", x.shape)
        x = self.pool(x)
        # print("pool:", x.shape)
        x = torch.flatten(x, 1)
        # print("flatten:", x.shape)
        return self.fc(x)
    

class myDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


X = torch.randn(30, 1, 28, 28)
y = torch.randint(0, 3, (30,))

dataset = myDataset(X,y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = op.SGD(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    for x_i, y_i in dataloader:
        
        optimizer.zero_grad()
        y_i_predict = model(x_i)
        loss = loss_fn(y_i_predict, y_i)
        loss.backward()
        optimizer.step()
        
        
model.eval()
with torch.no_grad():
    
    logits = model(X)
    ypred = torch.argmax(logits, dim=1)
    loss = loss_fn(logits,y)
    
    print(f'Test Loss {loss}')
    print(f'Predictions: {ypred} | logits: {logits}')