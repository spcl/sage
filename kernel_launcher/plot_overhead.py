import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import itertools
import scipy


entries = []

with open('perf_kernel_and_copy.txt', 'r') as f:
    for l in f.readlines():
        match = re.search('Start run protected (.+) :: args (.+) (.+) (.+) (.+) (.+) (.+)', l)
        if match:
            version = 'SAGE' if int(match.group(1)) else 'Baseline'
            batch = int(match.group(4)) * 128
            copy_repeats = int(match.group(5))
            relu_repeats = int(match.group(6))
    
        match = re.search('times_(.+) mean (.+) ms std (.+) ms', l)
        if match and match.group(1) == 'k1':
            mean = float(match.group(2))
            std = float(match.group(3))
            entries.append({
                'Batch': batch,
                'Operation': 'Kernel launch',
                'Version': version,
                'Mean': mean / relu_repeats,
                'Std': std / relu_repeats,
            })
        if match and match.group(1) == 'in':
            mean = float(match.group(2))
            std = float(match.group(3))
            entries.append({
                'Batch': batch,
                'Operation': 'Copy',
                'Version': version,
                'Mean': mean / copy_repeats,
                'Std': std / copy_repeats,
            })

df = pd.DataFrame.from_dict(entries)

print(df)

x_copy = np.array(df[(df.Operation == 'Copy') & (df.Version == 'SAGE')].Batch)
# convert to MB
x_copy = x_copy * 784 * 4 / 1e9

x_kernel = np.array(df[(df.Operation == 'Kernel launch') & (df.Version == 'SAGE')].Batch)

val = {}
for op, ver in itertools.product(df.Operation.unique(), df.Version.unique()):
    val[(op, ver, 'mean')] = np.array(df[(df.Operation == op) & (df.Version == ver)].Mean)
    val[(op, ver, 'std')] = np.array(df[(df.Operation == op) & (df.Version == ver)].Std)

diff = {}
for op in df.Operation.unique():
    diff[(op, 'mean')] = val[(op, 'SAGE', 'mean')] - val[(op, 'Baseline', 'mean')]
    diff[(op, 'std')] = val[(op, 'SAGE', 'std')] + val[(op, 'Baseline', 'std')]

k_lin = scipy.stats.linregress(x_kernel, diff['Kernel launch', 'mean'])
c_lin = scipy.stats.linregress(x_copy, diff['Copy', 'mean'])

x_copy_approx = np.array([x_copy[0], x_copy[-1]])
y_copy_approx = np.array([c_lin.intercept + c_lin.slope * x for x in x_copy_approx])

sns.set_theme()

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

ax[0].errorbar(x_copy, diff['Copy', 'mean'], yerr=diff['Copy', 'std'], label='measured')
#ax[0].axline((0, c_lin.intercept), slope=c_lin.slope, label='approximated')
ax[0].plot(x_copy_approx, y_copy_approx, label=f'{c_lin.intercept:.2f} + {c_lin.slope:.2f} * x')
ax[0].set_xlabel('Input size [GB]')
ax[0].set_ylabel('Time [ms]')
ax[0].set_title('Copy')
ax[0].legend()

#ax[1].errorbar(x_kernel, diff['Kernel launch', 'mean'], yerr=diff['Kernel launch', 'std'])

# ax[1].errorbar(x_kernel, val[('Kernel launch', 'SAGE', 'mean')], yerr=diff['Kernel launch', 'std'], label='SAGE')
# ax[1].errorbar(x_kernel, val[('Kernel launch', 'Baseline', 'mean')], yerr=diff['Kernel launch', 'std'], label='Baseline')

kernel_time = val[('Kernel launch', 'Baseline', 'mean')]
kernel_std = val[('Kernel launch', 'Baseline', 'std')]
launch_percent = 100 * diff['Kernel launch', 'mean'] / val[('Kernel launch', 'Baseline', 'mean')]

ax[1].errorbar(kernel_time, launch_percent, marker='.')

min_time = kernel_time[launch_percent < 5][0]
min_percent = launch_percent[launch_percent < 5][0]

ax[1].annotate(
    f"{kernel_time[0]:.2f} ms", 
    xy=(kernel_time[0], launch_percent[0]), 
    xytext=(20, -20),
    textcoords='offset points',
    arrowprops=dict(facecolor='black', shrink=0.05),
)

ax[1].annotate(
    f"{min_time:.2f} ms", 
    xy=(min_time, min_percent), 
    xytext=(20, 20),
    textcoords='offset points',
    arrowprops=dict(facecolor='black', shrink=0.05),
)

#ax[1].legend()
ax[1].set_xlabel('Kernel runtime [ms]')
ax[1].set_title('Kernel')
ax[1].set_ylabel('Launch overhead [%]')
# ax[1].set_xlabel('Batch size [1e3]')
# ax[1].set_title('Kernel launch')

fig.tight_layout()

plt.savefig('perf_overhead.png', bbox_inches='tight', pad_inches=0.0)
plt.savefig('perf_overhead.pdf', bbox_inches='tight', pad_inches=0.0)
