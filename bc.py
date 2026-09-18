import torch
import torch.nn as nn
import torch.optim as op
from sklearn.metrics import confusion_matrix


X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])

# print(X[:5], y[:5])

class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1,1)
        
    def forward(self,x):
        return self.linear_layer(x)

epochs = 500

model = LinearRegression()
optimizer = op.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    
    model.train()
    optimizer.zero_grad()
    
    y_predict = model(X)
    
    # loss_fn = nn.L1Loss(y, y_predict)
    loss = loss_fn(y_predict, y)
    
    # if epoch%10 == 0:
    #     print(f'Loss: {loss} | predicted value: {y_predict} | actual value: {y}')
    
    loss.backward()
    optimizer.step()
    
model.eval()

X_test = torch.tensor([
    [1.5],
    [3.0],
    [4.2],
    [5.5],
    [7.0],
    [8.5],
])

y_test = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
])

with torch.no_grad():
    y_logists = model(X_test)
    y_prob = torch.sigmoid(y_logists)
    print(f'Probaility on logists: {y_prob} | logits: {y_logists}')
    y_predict = (y_prob >= 0.5).float()
    print(f'Accuracy of model is {(y_test == y_predict).float().mean()}')
    
    print(confusion_matrix(y_pred=y_predict,y_true=y_test))