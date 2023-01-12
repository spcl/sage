import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import itertools


entries = []

with open('perf.txt', 'r') as f:
    for l in f.readlines():
        match = re.search('Start run protected (.+) :: args (.+) (.+) (.+) (.+) (.+) (.+)', l)
        if match:
            version = 'Protected' if int(match.group(1)) else 'Base'
            batch = int(match.group(4)) * 128
    
        match = re.search('times mean (.+) ms std (.+) ms', l)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            entries.append({
                'Batch': batch,
                'Version': version,
                'Mean': mean * 1e3 / batch,
                'Std': std * 1e3 / batch,
            })

df = pd.DataFrame.from_dict(entries)

print(df)

sns.set_theme()

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

width = 0.4
num_groups = len(df.Version.unique())
num_x_elems = len(df) // num_groups
for idx, ver in enumerate(df.Version.unique()):
    x = [(x + width * (idx - (num_groups - 1) / 2)) for x in range(num_x_elems)]
    y = df[df.Version == ver].Mean
    yerr = df[df.Version == ver].Std 
    ax.bar(x, y, yerr=yerr, width=width, label=ver)
        
ax.set_xticks(range(len(df.Batch.unique())))
ax.set_xticklabels(df.Batch.unique(), minor=False)
ax.legend(ncol=2, fontsize='small')
ax.set_ylabel('Time per sample [us]')
ax.set_xlabel('Batch size')

fig.tight_layout()

plt.savefig('perf.png', bbox_inches='tight', pad_inches=0.0)
plt.savefig('perf.pdf', bbox_inches='tight', pad_inches=0.0)
