import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import scienceplots

class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()
        self.hparams = hparams
        input = hparams["input"]
        nodes = hparams["nodes"]
        layers = hparams["layers"]
        output = hparams["output"]


        net = [nn.Linear(input, nodes), nn.ReLU()]
        for _ in range(layers-1):
            net.extend([nn.Linear(nodes, nodes), nn.ReLU()])
        net.append(nn.Linear(nodes, output))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# Set seed
torch.manual_seed(42)

# Create Data
x = torch.rand(200, 1)
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn_like(x)

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.autoscale(tight=True)
    fig.savefig("mlp_data.png", dpi=600, bbox_inches="tight")

# Create Model
hparams = {
    "input": 1,
    "nodes": 16,
    "layers": 3,
    "output": 1
}
model = MLP(hparams)

# Test Model
y_pred = model(x)
loss = F.mse_loss(y_pred, y)
print(loss)

# Set Hyperparameters
lr = 0.1
epochs = 3000

# Train Model
optimizer = SGD(model.parameters(), lr=lr)
loss_vec = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)
    loss_vec.append(loss.item())
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Plot Loss
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(loss_vec)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.autoscale(tight=True)
    fig.savefig("mlp_loss.png", dpi=600, bbox_inches="tight")

# Plot Predictions
y_hat = model(x)
x_pred = torch.linspace(0, 1, 200).unsqueeze(1)
y_pred = model(x_pred)
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="True")
    ax.plot(x_pred, y_pred.detach(), label="Predicted", color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Loss: {F.mse_loss(y_hat, y).item():.4e}")
    ax.autoscale(tight=True)
    ax.legend()
    fig.savefig("mlp_preds.png", dpi=600, bbox_inches="tight")
