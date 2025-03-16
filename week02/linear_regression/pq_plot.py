import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import argparse

parser = argparse.ArgumentParser(description='Plot parquet file')
parser.add_argument('--trial', type=int, default=100, help='Trial number')
parser.add_argument('--eta', type=float, default=0.1, help='Learning rate')
args = parser.parse_args()
filename = f'results/linreg_{args.trial}_{args.eta}.parquet'
trial = args.trial
eta = args.eta

# Import parquet file
df_data = pd.read_parquet("results/linreg_data.parquet")
df = pd.read_parquet(filename)

# Prepare Data to Plot
x_data = np.array(df_data['x'][:])
y_data = np.array(df_data['y'][:])
w0 = np.array(df['w0'][:])
w1 = np.array(df['w1'][:])
loss = np.array(df['loss'][:])

x = np.linspace(0, 1, 100)
ys = []
for i in range(len(w0)):
    ys.append(w0[i] + w1[i]*x)
ys = np.array(ys)

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)

    ax.plot(x_data, y_data, '.', label='Data')
    for i in range(len(w0) // 5, len(w0), len(w0) // 5):
        ax.plot(x, ys[i], label=f'{i}th trial', alpha=0.5)
    ax.plot(x, ys[-1], label='Final', color='black', alpha=0.5)
    ax.legend()
    fig.savefig(f'figs/linreg_{trial}_{eta}.png', dpi=600, bbox_inches='tight')

# Loss Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.plot(loss, label='Loss')
    ax.legend()
    fig.savefig(f'figs/loss_{trial}_{eta}.png', dpi=600, bbox_inches='tight')
