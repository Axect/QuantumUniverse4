import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR, CosineAnnealingLR
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary # For model summary
import argparse

# ┌──────────────────────────────────────────────────────────┐
#  0. Configuration & Device Setup
# └──────────────────────────────────────────────────────────┘
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the number of data points
num_data_points = 10000
# Define the batch size for mini-batch training
batch_size = 64 # You might want to experiment with this value


# ┌──────────────────────────────────────────────────────────┐
#  1. Data Generation
# └──────────────────────────────────────────────────────────┘
# Create a more complex 1D function for regression
# y = sin(x) * exp(-0.1*x) + 0.1 * cos(5*x) + noise
print("Generating complex 1D data...")
x_data = np.linspace(-5, 15, num_data_points) # Generate x values
y_true = np.sin(x_data) * np.exp(-0.1 * x_data) + 0.1 * np.cos(5 * x_data)
noise_level = 0.05
y_noisy = y_true + np.random.randn(num_data_points) * noise_level

# Convert NumPy arrays to PyTorch tensors
# Reshape tensors to have shape [num_data_points, 1]
# Note: Tensors are NOT moved to device here; batches will be moved in the training loop.
x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_noisy, dtype=torch.float32).unsqueeze(1)
y_true_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)

print(f"Generated {num_data_points} data points.")
print(f"Input tensor shape: {x_tensor.shape}")
print(f"Target tensor shape: {y_tensor.shape}")


# ┌──────────────────────────────────────────────────────────┐
#  2. Dataset and DataLoader
# └──────────────────────────────────────────────────────────┘
# Create a TensorDataset
dataset = TensorDataset(x_tensor, y_tensor)

# Create a DataLoader for mini-batching and shuffling (Fill this in)
# Refer to https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Created TensorDataset and DataLoader with batch size {batch_size}.")


# ┌──────────────────────────────────────────────────────────┐
#  3. Model Definition (Homework: Fill this in)
# └──────────────────────────────────────────────────────────┘
# Define your neural network model. It should inherit from nn.Module.
class RegressionModel(nn.Module):
    def __init__(self, hparams): # Accept hyperparameters if needed
        super(RegressionModel, self).__init__()
        # HINT: Define the necessary layers (nn.Linear, activations like nn.ReLU, etc.)
        #       Input size should be 1, output size should be 1.
        #       Consider using hparams dictionary if you want to pass node counts, layers etc.
        nodes = hparams["nodes"]
        layers = hparams["layers"]
        net = [nn.Linear(1, nodes), nn.GELU()]
        for _ in range(layers):
            net.extend([nn.Linear(nodes, nodes), nn.GELU()])
        net.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # HINT: Implement the forward pass, connecting the layers.
        output = self.net(x)
        return output


# ┌──────────────────────────────────────────────────────────┐
#  4. Training Setup (Homework: Fill this in)
# └──────────────────────────────────────────────────────────┘

parser = argparse.ArgumentParser()
parser.add_argument("--nodes", type=int, default=32, help="Number of nodes in each hidden layer")
parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--scheduler", type=str, default="R", help="Learning rate scheduler type")
args = parser.parse_args()

# HINT: Define hyperparameters (optional, but good practice)
# hparams = {
#     'nodes': ...,
#     'layers': ...,
# }
hparams = {
    "nodes": args.nodes,
    "layers": args.layers,    
} # YOUR CODE HERE (Define your hyperparameters)

# HINT: Instantiate the model defined above.
model = RegressionModel(hparams) # YOUR CODE HERE

# HINT: Print model summary (optional, requires torchinfo). Input size is (batch_size, input_features)
summary(model, input_size=(1, 1))

# HINT: Move the model to the designated device (e.g., GPU or CPU).
model.to(device)

# HINT: Define the loss function suitable for regression (e.g., nn.MSELoss).
criterion = nn.MSELoss() # YOUR CODE HERE

# HINT: Define the optimizer (e.g., optim.Adam). Link it to model.parameters().
optimizer = optim.Adam(model.parameters(), lr=2e-3) # YOUR CODE HERE

# Scheduler
scheduler = None
if args.scheduler == "R":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
elif args.scheduler == "P":
    scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=2.0)
elif args.scheduler == "C":
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
elif args.scheduler == "H":
    scheduler = HyperbolicLR(optimizer, upper_bound=600, max_iter=args.epochs, infimum_lr=1e-5)
elif args.scheduler == "EH":
    scheduler = ExpHyperbolicLR(optimizer, upper_bound=600, max_iter=args.epochs, infimum_lr=1e-5)


# ┌──────────────────────────────────────────────────────────┐
#  5. Training Loop (Homework: Fill this in)
# └──────────────────────────────────────────────────────────┘
epochs = args.epochs # Define the number of training epochs
print(f"\nStarting training for {epochs} epochs...")
# HINT: Create a list or array to store loss per epoch for plotting later.
epoch_losses = []
lrs = []

# HINT: Loop through the epochs.
for epoch in range(epochs):
    # Keep track of loss within the epoch if needed.
    running_loss = 0.0

    # Loop through the DataLoader (`train_loader`).
    for batch_x, batch_y in train_loader:
        # Move the current batch of data to the device.
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # --- Mini-batch training steps ---
        # 1. HINT: Zero the gradients
        optimizer.zero_grad()

        # 2. HINT: Perform the forward pass (get model predictions).
        # outputs = ...
        outputs = model(batch_x)

        # 3. HINT: Calculate the loss.
        # loss = ...
        loss = criterion(outputs, batch_y)
        
        # 4. HINT: Perform the backward pass
        loss.backward()
        
        # 5. HINT: Update the weights
        optimizer.step()

        # HINT: Accumulate the loss for the epoch
        # running_loss += loss.item() * batch_x.size(0) # Weighted by batch size
        running_loss += loss.item() * batch_x.size(0)

    # 6. Optional: Step the scheduler if using one
    if scheduler is not None:
        if args.scheduler == "R":
            scheduler.step(loss)
        else:
            scheduler.step()

    # Calculate and store the average loss for the epoch.
    epoch_loss = running_loss / len(dataset)
    epoch_losses.append(epoch_loss)

    lrs.append(optimizer.param_groups[0]['lr']) # Store learning rate for plotting

    # Print epoch loss periodically.
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4e}, lr: {optimizer.param_groups[0]["lr"]:.4e}')

print("Training finished.")


# ┌──────────────────────────────────────────────────────────┐
#  6. Evaluation & Visualization (Optional Homework)
# └──────────────────────────────────────────────────────────┘
print("\nEvaluating model...")
# HINT: Set the model to evaluation mode.
model.eval()

# HINT: Use torch.no_grad() context manager for inference.
with torch.no_grad():
    # HINT: Generate predictions for the *entire* dataset (x_tensor).
    #       Remember to move x_tensor to the device for prediction.
    predicted = model(x_tensor.to(device))

true_loss = criterion(predicted, y_true_tensor.to(device))

# HINT: Move predictions back to CPU and convert to NumPy for plotting.
y_pred_numpy = predicted.cpu().numpy()

# --- Plotting Results ---
# HINT: Use matplotlib to plot:
#       - Noisy data (scatter)
#       - True function (line)
#       - Model predictions (line)
#       Use x_data, y_noisy, y_true, y_pred_numpy for plotting.
#       Add labels, title, legend, grid.
#       Save the plot using plt.savefig("regression_results.png")

# Example plotting structure (uncomment and adapt after training):
plt.figure(figsize=(12, 7))
plt.scatter(x_data, y_noisy, label='Noisy Data', alpha=0.1, s=10, color='blue')
plt.plot(x_data, y_true, label='True Function', color='green', linewidth=2)
plt.plot(x_data, y_pred_numpy, label='NN Prediction', color='red', linestyle='--', linewidth=2) # Use your prediction variable
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Loss between True and Predicted: {true_loss.item():.4e}")
plt.legend()
plt.grid(True)
plt.ylim(y_true.min() - 1, y_true.max() + 1)
plt.savefig(f"fig_schedulers/regression_{hparams["nodes"]}_{hparams["layers"]}_{epochs}_{args.scheduler}_GELU.png", dpi=300, bbox_inches='tight')
# plt.show()
# print("Plotting results (placeholder).")


# ┌──────────────────────────────────────────────────────────┐
#  7. Loss Plot (Optional Homework)
# └──────────────────────────────────────────────────────────┘
# HINT: Plot the `epoch_losses` collected during training.
#       Use a logarithmic scale for the y-axis if loss decreases rapidly (plt.yscale('log')).
#       Add labels, title, legend, grid.
#       Save the plot using plt.savefig("loss_curve.png")

# Example plotting structure (uncomment and adapt after training):
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale('log') # Optional: use log scale if loss drops quickly
plt.ylim((2e-3, 2e-1))
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"fig_schedulers/loss_curve_{hparams["nodes"]}_{hparams["layers"]}_{epochs}_{args.scheduler}_GELU.png", dpi=300, bbox_inches='tight')
# plt.show()
# print("Plotting loss curve (placeholder).")
