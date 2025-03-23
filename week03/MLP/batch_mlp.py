import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import scienceplots
import argparse

# Define Model
class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()
        self.hparams = hparams
        input = hparams["input"]
        nodes = hparams["nodes"]
        layers = hparams["layers"]
        output = hparams["output"]


        net = [nn.Linear(input, nodes), nn.ReLU()]
        for _ in range(layers):
            net.extend([nn.Linear(nodes, nodes), nn.ReLU()])
        net.append(nn.Linear(nodes, output))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# Set seed
torch.manual_seed(42)

# Create Data
x = torch.rand(1000, 1)
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn_like(x)
ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=32, shuffle=True)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", type=int, default=16)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=300)
args = parser.parse_args()

# Create Model
hparams = {
    "input": 1,
    "nodes": args.nodes,
    "layers": args.layers,
    "output": 1
}
model = MLP(hparams)

# Test Model
y_pred = model(x)
loss = F.mse_loss(y_pred, y)
print(loss)

# Set Hyperparameters
lr = args.lr
epochs = args.epochs

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x_gpu = x.to(device)
y_gpu = y.to(device)
optimizer = SGD(model.parameters(), lr=lr)
loss_vec = []
for epoch in range(epochs):
    loss_epoch = 0
    for x_batch, y_batch in dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_hat = model(x_batch)
        loss = F.mse_loss(y_hat, y_batch)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    loss_vec.append(loss_epoch / len(dl))
    if epoch % 10 == 0:
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
model.eval()
model.to("cpu")
y_hat = model(x)
x_pred = torch.linspace(0, 1, 1000).unsqueeze(1)
y_pred = model(x_pred)
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', markersize=3, markeredgewidth=0, label="Data")
    ax.plot(x_pred, y_pred.detach(), label="Predicted", color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Loss: {F.mse_loss(y_hat, y).item():.4e}")
    ax.autoscale(tight=True)
    ax.legend()
    fig.savefig("mlp_preds.png", dpi=600, bbox_inches="tight")
