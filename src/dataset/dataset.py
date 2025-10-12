import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict


class SinhalaCharDataset(Dataset):
    """Enhanced dataset for Sinhala handwritten characters with tablet optimization"""

    def __init__(self, root_dir, transform=None, cache_images=False):
        """
        Args:
            root_dir (str): Path to data folder (train/val/test)
            transform: Optional transforms to apply
            cache_images (bool): Cache images in memory for faster training
        """
        # Convert to absolute path to avoid path resolution issues
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_counts = defaultdict(int)
        self.image_cache = {} if cache_images else None

        print(f"📁 Dataset root: {self.root_dir}")
        print(f"📁 Root exists: {os.path.exists(self.root_dir)}")
        print(f"💾 Image caching: {'enabled' if cache_images else 'disabled'}")

        # Scan folders 1-454 and collect image paths
        self._load_samples()

    def _load_samples(self):
        """Scan all class folders and collect image paths with labels"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for class_id in range(1, 455):  # folders 1 to 454
            folder_name = str(class_id)
            folder_path = os.path.join(self.root_dir, folder_name)

            if not os.path.exists(folder_path):
                continue

            # Map folder to index (folder 1 -> index 0, folder 454 -> index 453)
            idx = class_id - 1
            self.class_to_idx[folder_name] = idx
            self.idx_to_class[idx] = folder_name

            # Collect all images in this folder
            image_files = []
            for img_name in os.listdir(folder_path):
                if any(img_name.lower().endswith(ext) for ext in valid_extensions):
                    img_path = os.path.join(folder_path, img_name)
                    if os.path.isfile(img_path):  # Ensure it's a file
                        image_files.append(img_path)

            # Add samples and count
            for img_path in image_files:
                self.samples.append((img_path, idx))
                self.class_counts[idx] += 1

        print(f"📊 Loaded {len(self.samples):,} samples from {len(self.class_to_idx)} classes")

        # Print class distribution stats
        if self.class_counts:
            counts = list(self.class_counts.values())
            print(f"📊 Samples per class - Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get sample with enhanced error handling"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        img_path, label = self.samples[idx]

        try:
            # Load from cache or disk
            if self.image_cache is not None and img_path in self.image_cache:
                image = self.image_cache[img_path]
            else:
                image = Image.open(img_path)

                # Ensure RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Cache if enabled
                if self.image_cache is not None:
                    self.image_cache[img_path] = image

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback_image = Image.new('RGB', (112, 112), color='black')
                return self.transform(fallback_image), label
            else:
                return torch.zeros(3, 112, 112), label

    @property
    def num_classes(self):
        """Return number of classes"""
        return len(self.class_to_idx)

    def get_class_weights(self):
        """Calculate class weights for handling imbalanced dataset"""
        if not self.class_counts:
            return None

        # Calculate inverse frequency weights
        total_samples = sum(self.class_counts.values())
        num_classes = len(self.class_counts)

        weights = []
        for class_idx in range(num_classes):
            if class_idx in self.class_counts:
                weight = total_samples / (num_classes * self.class_counts[class_idx])
                weights.append(weight)
            else:
                weights.append(1.0)  # Default weight for missing classes

        return torch.FloatTensor(weights)

    def get_class_distribution(self):
        """Get detailed class distribution"""
        return dict(self.class_counts)

    def get_samples_for_class(self, class_idx):
        """Get all sample paths for a specific class"""
        return [path for path, label in self.samples if label == class_idx]


class TabletCanvasDataset(Dataset):
    """Specialized dataset for tablet canvas input processing"""

    def __init__(self, canvas_images, transform=None):
        """
        Args:
            canvas_images: List of PIL Images from tablet canvas
            transform: Transforms optimized for tablet input
        """
        self.canvas_images = canvas_images
        self.transform = transform

    def __len__(self):
        return len(self.canvas_images)

    def __getitem__(self, idx):
        image = self.canvas_images[idx]

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, -1  # No label for inference


def create_balanced_sampler(dataset):
    """Create a balanced sampler for handling class imbalance"""
    from torch.utils.data import WeightedRandomSampler

    # Calculate sample weights
    class_counts = dataset.get_class_distribution()
    sample_weights = []

    for _, label in dataset.samples:
        # Inverse frequency weighting
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
