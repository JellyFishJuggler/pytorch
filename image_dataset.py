import torch
import torch.nn as nn
import torch.optim as op
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pandas as py
from pathlib import Path

import time
start = time.time()

device = torch.device("cuda")

IMG_PATH = './data/FETAL_PLANES_ZENODO/Images/Patient00001_Plane1_2_of_15.png' 
# Patient00001_Plane1_1_of_15.png
IMG_DIR = './data/FETAL_PLANES_ZENODO/Images'
image = Image.open(IMG_PATH)

df = py.read_csv('data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv',sep=';')

class myDataset(Dataset):
    
    def __init__(self,df,imgDIR,transform):
        super().__init__()
        self.df = df
        self.imgDIR = imgDIR
        self.transform = transform
        self.class_to_indx = {
            "Other" : 0,
            "Fetal brain" : 1,
            "Fetal thorax" : 2,
            "Maternal cervix" : 3,
            "Fetal femur" : 4,
            "Fetal abdomen" : 5
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row['Image_name'] + '.png'
        base_dir = Path('./data/FETAL_PLANES_ZENODO/Images')
        img_path = base_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(image)
        
        label = self.class_to_indx[row['Plane']]

        return img, label

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32*54*54,6)
        )
    
    def forward(self,x):
        return self.network(x)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_df = df[df['Train '] == 1]
test_df = df[df['Train '] == 0]

train_dataset = myDataset(train_df,IMG_DIR,transform)
test_dataset = myDataset(test_df,IMG_DIR,transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNN()
model.to(device)
# images, labels = next(iter(train_dataloader))       #batches
loss_fn = nn.CrossEntropyLoss()
optimizer = op.SGD(model.parameters(),lr=0.01)

epochs = 10
for epoch in range(epochs):
    for x_i,y_i in train_dataloader:
        
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        
        optimizer.zero_grad()
        predict = model(x_i)
            
        loss = loss_fn(predict,y_i)
        loss.backward()
        optimizer.step()
        print("Training time:", time.time() - start)

correct = 0
total = 0
test_loss = 0

model.eval()
with torch.no_grad():
    for x_i,y_i in test_dataloader:
        
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        
        logits = model(x_i)
        predict = torch.argmax(logits,dim=1)
        loss = loss_fn(logits,y_i)
        
        correct += (predict == y_i).float().sum()
        total += len(y_i)
        test_loss += loss.item()
        
test_loss /= len(test_dataloader)
accuracy = correct/total

print("Test Loss:", test_loss)
print("Accuracy:", accuracy.item())