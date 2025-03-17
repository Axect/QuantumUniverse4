# Import
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

EPOCHS = 500

# Set Seed
torch.manual_seed(42)

# Define Model
class LinReg(nn.Module):
    def __init__(self):
        super(LinReg, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)

# Create Data
x = torch.rand(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

# Create Model
model = LinReg()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
loss_vec = np.zeros(EPOCHS)

# Training
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    loss_vec[epoch] = loss.item()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')

# Loss Plot
with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots()
    ax.plot(loss_vec)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.autoscale(tight=True)
    fig.savefig('loss.png', dpi=600, bbox_inches='tight')

# Test Plot
with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', label='Data')
    ax.plot(x, model(x).detach().numpy(), label='Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.savefig('test.png', dpi=600, bbox_inches='tight')
