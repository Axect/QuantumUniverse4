import torch
from torch import nn
from torch.nn import functional as F

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


# Create Data
x = torch.rand(100, 1)
y = 2 * (x - 0.5) ** 2 + 0.1 * torch.randn(100, 1)

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
