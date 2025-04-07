# Import the pytorch library
import torch

# ┌──────────────────────────────────────────────────────────┐
#  Simple Harmonic Oscillator
# └──────────────────────────────────────────────────────────┘
# 1. Define constants
k = torch.tensor(2.0) # spring constant (N/m)
m = torch.tensor(0.5) # mass (kg)

# 2. Define variables
# We want to calculate gradients with respect to the variables,
# so we need to set `requires_grad=True`
x = torch.tensor(1.5, requires_grad=True) # Initial position (m)
v = torch.tensor(0.8, requires_grad=True) # Initial velocity (m/s)

# 3. Define the potential energy function
V = 0.5 * k * x**2 # Potential energy (J)
print(f"Position (x): {x.item():.4f} [m]")
print(f"Velocity (v): {v.item():.4f} [m/s]")
print(f"Potential Energy (V): {V.item():.4f} [J]")

# 4. Obtain the gradient of V w.r.t x (dV/dx)
V.backward() # Compute dV/dx and stores it in x.grad

# 5. Access the gradient (Force)
F = -x.grad
print(f"\nForce (F = -dV/dx): {F.item():.4f} [N]")
print(f"Force (Analytic): {(-k * x).item():.4f} [N]")

# 6. Kinetic Energy
x.grad.zero_() # Reset the gradient (if not, it will accumulate)

T = 0.5 * m * v**2 # Kinetic energy (J)
print(f"\nKinetic Energy (T): {T.item():.4f} [J]")

T.backward() # Compute dT/dv and stores it in v.grad

# 7. Access the gradient (Momentum)
P = v.grad
print(f"Momentum (P = dT/dv): {P.item():.4f} [kg*m/s]")
print(f"Momentum (Analytic): {(m * v).item():.4f} [kg*m/s]")

# 8. Total Energy
# Since we already called backward on T and V, the graph was freed by default.
# To compute dE/dx and dE/dv, we would need to define E *before* calling backward,
# or use retain_graph=True in the backward call.

# Re-define the variables
x = torch.tensor(1.5, requires_grad=True)
v = torch.tensor(0.8, requires_grad=True)

V = 0.5 * k * x**2
T = 0.5 * m * v**2
E = T + V # Total energy (J)

print(f"\nTotal Energy (E): {E.item():.4f} [J]")

E.backward() # Compute dE/dx and dE/dv

print(f"\nForce (F = -dE/dx): {-x.grad.item():.4f} [N]")
print(f"Momentum (P = dE/dv): {v.grad.item():.4f} [kg*m/s]")
