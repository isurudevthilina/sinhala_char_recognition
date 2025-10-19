import os
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(ROOT, 'data', 'train')
out_dir = os.path.join(ROOT, 'evaluation_results')
out_path = os.path.join(out_dir, 'sample_grid_4x4.png')
grid_size = (4, 4)
cell_size = (160, 160)  # width, height

# Helpers
def find_image_in_dir(d):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for f in os.listdir(d):
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(d, f)):
            return os.path.join(d, f)
    return None

# Collect class directories
class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x) if x.isdigit() else x)
if len(class_dirs_sorted) == 0:
    raise SystemExit('No class folders found in data/train')

# Choose 16 classes evenly spaced across sorted list
num_cells = grid_size[0] * grid_size[1]
indices = []
if len(class_dirs_sorted) >= num_cells:
    step = len(class_dirs_sorted) / num_cells
    indices = [int(i * step) for i in range(num_cells)]
else:
    # if fewer classes than cells, pick all and repeat some
    indices = list(range(len(class_dirs_sorted)))
    while len(indices) < num_cells:
        indices.extend(indices[:(num_cells - len(indices))])

selected_dirs = [class_dirs_sorted[i] for i in indices]

# Load images for each selected class
images = []
labels = []
for d in selected_dirs:
    p = os.path.join(train_dir, d)
    img_path = find_image_in_dir(p)
    if img_path is None:
        # create blank placeholder
        img = Image.new('RGB', cell_size, color=(240,240,240))
        images.append(img)
        labels.append(d)
        continue
    img = Image.open(img_path).convert('RGB')
    img = img.resize(cell_size, Image.LANCZOS)
    images.append(img)
    labels.append(d)

# Create grid figure
fig_w = grid_size[1] * (cell_size[0]/100)
fig_h = grid_size[0] * (cell_size[1]/100)
fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8,8))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

idx = 0
for r in range(grid_size[0]):
    for c in range(grid_size[1]):
        ax = axes[r, c]
        ax.imshow(images[idx])
        ax.axis('off')
        ax.set_title(f'Class {labels[idx]}', fontsize=9)
        idx += 1

# Ensure output dir
os.makedirs(out_dir, exist_ok=True)
plt.suptitle('Sample characters (4x4 grid)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_path, dpi=200)
print(f'Saved grid to: {out_path}')

