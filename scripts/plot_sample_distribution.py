import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt, pi, exp

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(ROOT, 'data', 'train')
out_dir = os.path.join(ROOT, 'evaluation_results')
out_path = os.path.join(out_dir, 'sample_distribution.png')

# Collect counts per class
class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
counts = []
for d in sorted(class_dirs, key=lambda x: int(x) if x.isdigit() else x):
    p = os.path.join(train_dir, d)
    # count image files (basic filter)
    files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    counts.append(len(files))

counts = np.array(counts)
if counts.size == 0:
    raise SystemExit('No class folders or samples found in data/train')

# Histogram
plt.figure(figsize=(9,6))
bins = max(20, min(100, int(len(counts)/2)))
_, bins_edges, _ = plt.hist(counts, bins=bins, color='#4c72b0', alpha=0.8, edgecolor='black')

# Overlay normal "bell curve" centered at 192
mu = 192.0
sigma = counts.std() if counts.std() > 0 else 20.0
x = np.linspace(bins_edges[0], bins_edges[-1], 400)
# Normal pdf
pdf = (1.0/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
# Scale pdf to histogram
bin_width = bins_edges[1] - bins_edges[0]
pdf_scaled = pdf * counts.size * bin_width
plt.plot(x, pdf_scaled, color='#dd8452', linewidth=2.5, label=f'Normal curve (mu={mu}, sigma={sigma:.1f})')

# Mean and median lines
plt.axvline(counts.mean(), color='k', linestyle='--', linewidth=1.2, label=f'Mean ({counts.mean():.1f})')
plt.axvline(np.median(counts), color='gray', linestyle=':', linewidth=1.2, label=f'Median ({np.median(counts):.1f})')

plt.title('Sample Distribution per Class (train split)')
plt.xlabel('Number of samples per class')
plt.ylabel('Number of classes')
plt.legend()
plt.grid(alpha=0.25)

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f'Saved histogram to: {out_path}')

