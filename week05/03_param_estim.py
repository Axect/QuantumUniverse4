import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

# 0. Set random seed & device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ┌──────────────────────────────────────────────────────────┐
#  Data Generation
# └──────────────────────────────────────────────────────────┘
# Simulate data from a physical process
# y(t) = y0 + v0 * t + 0.5 * g * t^2

# True parameters
g_true = -9.81      # gravity acceleration (m/s^2)
v0_true = 5.0       # initial velocity (m/s)
y0_true = 2.0       # initial position (m)
noise_level = 1.0   # Standard deviation of noise

# Generate time points
t_data = torch.linspace(0, 3, 100)

# Calculate true y values
y_data = y0_true + v0_true * t_data + 0.5 * g_true * t_data**2

# Add noise
y_noisy = y_data + torch.randn_like(y_data) * noise_level

# Reshape data for PyTorch
t_train = t_data.unsqueeze(1).to(device)   # Shape: (100, 1)
y_train = y_noisy.unsqueeze(1).to(device)  # Shape: (100, 1)

# ┌──────────────────────────────────────────────────────────┐
#  Define the Model
# └──────────────────────────────────────────────────────────┘
class GravityModel(nn.Module):
    def __init__(self):
        super(GravityModel, self).__init__()
        self.y0 = nn.Parameter(torch.tensor(0.0))
        self.v0 = nn.Parameter(torch.tensor(0.0))
        self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, t):
        return self.y0 + self.v0 * t + 0.5 * self.g * t**2

# ┌──────────────────────────────────────────────────────────┐
#  Training the Model
# └──────────────────────────────────────────────────────────┘
# Initialize model
model = GravityModel()
summary(model, input_size=(1, 1))

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
model = model.to(device)
epochs = 2000
losses = np.zeros(epochs)
print("\nTraining the model...")
for epoch in range(epochs):
    # 1) Forward pass
    y_pred = model(t_train)

    # 2) Compute loss
    loss = criterion(y_pred, y_train)

    # 3) Zero gradients
    optimizer.zero_grad()

    # 4) Backward pass
    loss.backward()

    # 5) Optimize
    optimizer.step()

    # 6) Store loss
    losses[epoch] = loss.item()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# ┌──────────────────────────────────────────────────────────┐
#  Evaluation & Visualization
# └──────────────────────────────────────────────────────────┘
# Set the model to evaluation mode
model.eval()

# Generate predictions
with torch.no_grad():
    y_pred = model(t_train)

# Convert tensors to numpy for plotting
t_numpy = t_data.numpy()
y_noisy_numpy = y_noisy.numpy()
y_true_numpy = y_data.numpy()
y_pred_numpy = y_pred.cpu().squeeze().numpy()

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(t_numpy, y_noisy_numpy, label='Training Data', alpha=0.6, s=20)
plt.plot(t_numpy, y_true_numpy, label='True Function', color='green', linewidth=2)
plt.plot(t_numpy, y_pred_numpy, label='Model Prediction', color='red', linestyle='--', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title(f'True Params: y0={y0_true:.2f}, v0={v0_true:.2f}, g={g_true:.2f} | Predicted Params: y0={model.y0.item():.2f}, v0={model.v0.item():.2f}, g={model.g.item():.2f}')
plt.legend()
plt.grid(True)
plt.savefig("parameter_plot.png", dpi=600)

# ┌──────────────────────────────────────────────────────────┐
#  Loss Plot
# └──────────────────────────────────────────────────────────┘
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("parameter_loss_plot.png", dpi=600)
