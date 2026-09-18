import torch
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,8,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(1352,3)
    
    def forward(self, x):
        x = self.conv(x)
        print("conv:", x.shape)
    
        x = self.relu(x)
        print("relu:", x.shape)
    
        x = self.pool(x)
        print("pool:", x.shape)
    
        x = torch.flatten(x, 1)
        print("flatten:", x.shape)
    
        return self.fc(x)


x = torch.randn(1, 1, 28, 28)
model = CNN()

y = model(x)
print(y)