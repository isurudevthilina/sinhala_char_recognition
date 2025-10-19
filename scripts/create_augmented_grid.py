import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(ROOT, 'data', 'train')
out_dir = os.path.join(ROOT, 'evaluation_results')
out_path = os.path.join(out_dir, 'augmented_grid_3x3.png')
cell_size = (200, 200)

# Helpers
def find_image_in_dir(d):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for f in os.listdir(d):
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(d, f)):
            return os.path.join(d, f)
    return None

class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x) if x.isdigit() else x)
if len(class_dirs_sorted) == 0:
    raise SystemExit('No class folders found in data/train')

# pick a representative class roughly in middle
idx = len(class_dirs_sorted)//2
selected_class = class_dirs_sorted[idx]
img_path = find_image_in_dir(os.path.join(train_dir, selected_class))
if img_path is None:
    # fallback to first class
    selected_class = class_dirs_sorted[0]
    img_path = find_image_in_dir(os.path.join(train_dir, selected_class))
    if img_path is None:
        raise SystemExit('No image files found in selected class folders')

orig = Image.open(img_path).convert('L')
orig = ImageOps.invert(orig) if np.mean(orig) > 200 else orig  # try to get darker ink on light bg
orig = orig.resize(cell_size, Image.LANCZOS)

# Augmentation functions
def rotate(img, angle):
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

def translate(img, tx, ty):
    return img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BICUBIC, fillcolor=255)

def shear(img, sx):
    return img.transform(img.size, Image.AFFINE, (1, sx, 0, 0, 1, 0), resample=Image.BICUBIC, fillcolor=255)

def scale_center(img, scale):
    w, h = img.size
    nw = int(w*scale)
    nh = int(h*scale)
    tmp = img.resize((nw, nh), Image.LANCZOS)
    # paste centered on white background
    bg = Image.new('L', img.size, color=255)
    x = (w - nw)//2
    y = (h - nh)//2
    bg.paste(tmp, (x, y))
    return bg

def adjust_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def gaussian_noise(img, sigma=10):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def gaussian_blur(img, radius=1.5):
    return img.filter(ImageFilter.GaussianBlur(radius))

# Define 8 augmentations
augments = [
    ('Rotate -15°', lambda im: rotate(im, -15)),
    ('Rotate +12°', lambda im: rotate(im, 12)),
    ('Translate (10, -8)', lambda im: translate(im, 10, -8)),
    ('Shear 0.25', lambda im: shear(im, 0.25)),
    ('Scale 0.9', lambda im: scale_center(im, 0.9)),
    ('Brightness 0.7', lambda im: adjust_brightness(im, 0.7)),
    ('Contrast 1.4', lambda im: adjust_contrast(im, 1.4)),
    ('Noise + Blur', lambda im: gaussian_blur(gaussian_noise(im, sigma=12), radius=1.2)),
]

# Prepare grid 3x3 with center original
grid_rows, grid_cols = 3, 3
cells = [None] * (grid_rows * grid_cols)
labels = [''] * (grid_rows * grid_cols)
# mapping positions: fill row-major; put original in center index 4
center_idx = 4
cells[center_idx] = orig
labels[center_idx] = f'Original (class {selected_class})'

# Apply augments to other positions
aug_idx = 0
for i in range(len(cells)):
    if i == center_idx:
        continue
    name, fn = augments[aug_idx]
    cells[i] = fn(orig)
    labels[i] = name
    aug_idx += 1
    if aug_idx >= len(augments):
        break

# Create matplotlib figure
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(6,6))
plt.subplots_adjust(wspace=0.4, hspace=0.6)
idx = 0
for r in range(grid_rows):
    for c in range(grid_cols):
        ax = axes[r, c]
        ax.imshow(cells[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(labels[idx], fontsize=8)
        idx += 1

os.makedirs(out_dir, exist_ok=True)
plt.suptitle('Original + 8 Augmented Variants', fontsize=12)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print('Saved augmented grid to:', out_path)

