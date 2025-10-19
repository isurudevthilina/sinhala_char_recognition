import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(ROOT, 'data', 'train')
out_dir = os.path.join(ROOT, 'evaluation_results')
out_path = os.path.join(out_dir, 'tsne_454_classes_by_category.png')

print('Script start')

# gather class folders
if not os.path.isdir(train_dir):
    raise SystemExit(f'data/train not found at expected path: {train_dir}')

class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x) if x.isdigit() else x)
if len(class_dirs_sorted) == 0:
    raise SystemExit('No class folders found in data/train')

print(f'Found {len(class_dirs_sorted)} class folders')

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
img_paths = []
labels = []
for d in class_dirs_sorted:
    p = os.path.join(train_dir, d)
    found = False
    for f in os.listdir(p):
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(p, f)):
            img_paths.append(os.path.join(p, f))
            labels.append(d)
            found = True
            break
    if not found:
        img_paths.append(None)
        labels.append(d)

n = len(img_paths)
print('Collected image paths for', n, 'classes')

# load class_map for characters
char_map = {}
class_map_path = os.path.join(ROOT, 'mappings', 'class_map.json')
if os.path.isfile(class_map_path):
    try:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_map = json.load(f)
        for k, v in class_map.items():
            if isinstance(v, dict):
                folder = v.get('folder_name')
                ch = v.get('character')
            else:
                # fallback formats
                folder = v
                ch = None
            if folder is not None:
                char_map[str(folder)] = ch
    except Exception as e:
        print('Warning: failed to read class_map.json:', e)
else:
    print('Warning: class_map.json not found; categories will be inferred from folder names')

# categorization helper
vowels = set(['අ','ආ','ඇ','ඈ','ඉ','ඊ','උ','එ','ඒ','ඔ','ඕ'])
modifier_chars = set(['ා','ැ','ෑ','ි','ී','ු','ූ','ෲ','්','ෝ'])

def categorize(char):
    if not char or not isinstance(char, str):
        return 'Unknown'
    if char in vowels:
        return 'Vowel'
    for ch in modifier_chars:
        if ch in char:
            return 'Modifier/Compound'
    if len(char) > 1:
        return 'Modifier/Compound'
    return 'Consonant'

categories = []
for lab in labels:
    ch = char_map.get(str(lab))
    if ch is None:
        # try using folder name if it contains non-numeric character
        ch = None
    categories.append(categorize(ch))

unique_cats = sorted(list(set(categories)))
print('Detected categories:', unique_cats)

# Feature extraction: use lightweight raw-pixel features to avoid heavy PyTorch/device issues
print('Using grayscale 80x80 raw pixels as features (fast, deterministic)')
arrs = []
for p in img_paths:
    if p is None:
        arrs.append(np.zeros((80,80), dtype=np.float32).flatten())
        continue
    try:
        im = Image.open(p).convert('L')
        # use Resampling.LANCZOS if available
        resample = getattr(Image, 'Resampling', None)
        if resample is not None:
            im = im.resize((80,80), resample.LANCZOS)
        else:
            im = im.resize((80,80), Image.LANCZOS)
        a = np.array(im).astype(np.float32).flatten()
        arrs.append(a)
    except Exception as e:
        print('Warning: failed to open', p, e)
        arrs.append(np.zeros((80,80), dtype=np.float32).flatten())
features = np.stack(arrs, axis=0)

print('Features matrix shape:', features.shape)

# Dimensionality reduction
pca_dim = min(50, features.shape[1])
if pca_dim < features.shape[1]:
    print('Running PCA to', pca_dim, 'dims')
    pca = PCA(n_components=pca_dim, random_state=42)
    feats_pca = pca.fit_transform(features)
else:
    feats_pca = features

print('Running t-SNE (this may take several minutes)...')
start = time.time()
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
emb = tsne.fit_transform(feats_pca)
end = time.time()
print(f't-SNE finished in {end-start:.1f}s')

# Plot
os.makedirs(out_dir, exist_ok=True)
plt.figure(figsize=(11,8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
cat_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_cats)}
for cat in unique_cats:
    idxs = [i for i,c in enumerate(categories) if c == cat]
    if len(idxs) == 0:
        continue
    pts = emb[idxs]
    plt.scatter(pts[:,0], pts[:,1], c=cat_to_color[cat], label=cat, s=25, alpha=0.85, edgecolors='none')

plt.title(f't-SNE embedding of {n} classes — color = category')
plt.xticks([])
plt.yticks([])
plt.legend(title='Category')
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print('Saved t-SNE plot to:', out_path)
