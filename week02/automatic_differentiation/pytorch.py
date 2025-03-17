import torch
from torchviz import make_dot

# Create a tensor
x = torch.tensor([1.], requires_grad=True)
u = x + 1
v = 2 * x ** 2
w = u + v

# Compute the gradient
w.backward(retain_graph=True)

# Visualize the computational graph
params = {'x': x, 'u': u, 'v': v, 'w': w}
dot = make_dot(w, params=params)

# Save the graph
dot.render('computational_graph', format='png')
