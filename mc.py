import torch
import torch.nn as nn
import torch.optim as op
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, train_size=0.8)

class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1,3)
        
    def forward(self,x):
        return self.linear_layer(x)
    

model = LinearRegression()
optimizer = op.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

epochs = 500

for epoch in range(epochs):
    
    model.train()
    optimizer.zero_grad()
    y_predict = model(X_train)
    loss = loss_fn(y_predict, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_logists = model(X_test)
    y_predict = torch.argmax(y_logists, dim=1)
    print(f'Predictions: {y_predict} | logits: {y_logists}')
    # y_predict = (y_prob >= 0.5).float()
    print(f'Accuracy of model is {(y_test == y_predict).float().mean()}')
    
    print(confusion_matrix(y_pred=y_predict,y_true=y_test))