import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Import parquet file
df = pd.read_parquet('kernel_regression.parquet')
dg = pd.read_parquet('kernel_true.parquet')

# Prepare Data to Plot
x_data = df['x']
y_pred1 = df['y1']
y_pred2 = df['y2']
y_pred3 = df['y3']
x_true = dg['x']
y_true = dg['y']

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
)

labels = ['Epanechnikov', 'Tri-cube', 'Gaussian']
colors = ['darkblue', 'darkred', 'darkgreen']
linestyles = ['--', '-.', ':']
ys = [y_pred1, y_pred2, y_pred3]
# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x_true, y_true, '.', label='Data', color='gray')
    for i, y in enumerate(ys):
        ax.plot(x_data, y, label=labels[i], color=colors[i],linewidth=1.5, alpha=0.6)
    ax.legend()
    fig.savefig('kernel_regression.png', dpi=600, bbox_inches='tight')
