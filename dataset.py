import torch
import torch.nn as nn
import torch.optim as op
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset, DataLoader


X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
])

y = torch.tensor([
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2,
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.8
)


class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,3)
        )
        # self.layer1 = nn.Linear(1, 8)
        # self.ReLU  = nn.ReLU()
        # self.layer2 = nn.Linear(8,3)
        

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.ReLU(x)
        # return self.layer2(x)
        return self.network(x)


class MyDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = MyDataset(X_train, y_train)

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True
)


print('Dataset Created')
print(dataset)
print(len(dataset))
print(dataset[1])


print('Dataloader Created')
for x_i, y_i in dataloader:
    print(x_i, y_i)

print('Training starting.')

model = LinearRegression()
optimizer = op.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


# x = torch.tensor([[5.0]])

# with torch.no_grad():
#     hidden = model.layer1(x)
#     print("Before ReLU:", hidden)

#     hidden_relu = model.Re(hidden)
#     print("After ReLU:", hidden_relu)

#     output = model.layer2(hidden_relu)
#     print("Final:", output)

epochs = 500
for epoch in range(epochs):

    epoch_loss = 0
    model.train()

    for x_i, y_i in dataloader:

        optimizer.zero_grad()

        y_predict = model(x_i)

        loss = loss_fn(y_predict, y_i)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1} | Loss {epoch_loss}")
model.eval()

print('Testing strting.')
with torch.no_grad():

    y_logists = model(X_test)
    y_predict = torch.argmax(y_logists, dim=1)
    test_loss = loss_fn(y_logists,y_test)
    
    print(f'Test Loss {test_loss}')
    print(f'Predictions: {y_predict} | logits: {y_logists}')

    print(
        f'Accuracy of model is '
        f'{(y_test == y_predict).float().mean()}'
    )

    print(
        confusion_matrix(
            y_pred=y_predict,
            y_true=y_test
        )
    )