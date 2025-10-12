from torchvision import transforms
import torch
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
import random


def get_train_transforms(img_size=80):  # Changed from 112 to 80
    """
    Tablet-optimized training transforms for paper-to-digital domain bridging
    Enhanced augmentations to simulate tablet stylus variations
    Optimized for 80x80 resolution to preserve image quality
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Advanced geometric augmentations for stylus variation
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.75, 1.3),
            shear=20
        ),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.6),

        # Tablet-specific stroke variations
        TabletStrokeSimulator(p=0.7),
        PressureVariation(p=0.6),
        StrokeWidthVariation(p=0.5),
        DigitalNoise(p=0.4),

        # Elastic deformation for natural stylus movement
        ElasticTransform(p=0.5, alpha=15, sigma=3),

        # Style and texture variations
        RandomBlur(p=0.4, max_radius=2.0),
        RandomSharpen(p=0.3),

        # Critical for domain adaptation - enhanced inversion
        SmartRandomInvert(p=0.6),

        # Advanced dropout augmentations
        CoarseDropout(p=0.3, max_holes=8, max_height=0.15, max_width=0.15),
        GridDistortion(p=0.4),

        # Color and contrast variations
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),

        transforms.ToTensor(),
        # Enhanced normalization for better convergence
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=80):  # Changed from 112 to 80
    """Validation transforms with tablet preprocessing simulation"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Light preprocessing to match tablet input
        TabletPreprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_inference_transforms(img_size=80):  # Changed from 112 to 80
    """
    Inference transforms optimized for tablet canvas input
    Handles various tablet drawing scenarios at 80x80 resolution
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        TabletCanvasProcessor(),  # Advanced tablet input processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Enhanced Custom Transforms for Tablet Optimization

class TabletStrokeSimulator:
    """Simulate tablet stylus stroke characteristics"""

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img.convert('L'))

        # Simulate stylus pressure tapering at stroke ends
        if torch.rand(1).item() > 0.5:
            # Add stroke tapering
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)

        # Simulate digital pen smoothing
        if torch.rand(1).item() > 0.6:
            img_np = cv2.bilateralFilter(img_np, 5, 25, 25)

        return Image.fromarray(img_np).convert('RGB')


class PressureVariation:
    """Simulate stylus pressure variations"""

    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img.convert('L'))

        # Create pressure map
        h, w = img_np.shape
        pressure_map = np.random.beta(2, 5, (h, w))
        pressure_map = gaussian_filter(pressure_map, sigma=2)

        # Apply pressure variation to strokes
        stroke_mask = img_np < 128
        img_np[stroke_mask] = (img_np[stroke_mask] * pressure_map[stroke_mask]).astype(np.uint8)

        return Image.fromarray(img_np).convert('RGB')


class StrokeWidthVariation:
    """Advanced stroke width variation for tablet simulation"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img.convert('L'))

        # Random morphological operations
        operations = [
            (cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)),
            (cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)),
            (cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
            (cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ]

        op, kernel = random.choice(operations)
        img_np = cv2.morphologyEx(img_np, op, kernel, iterations=1)

        return Image.fromarray(img_np).convert('RGB')


class DigitalNoise:
    """Add digital canvas noise and artifacts"""

    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img)

        # Add slight quantization noise (tablet digitization)
        if torch.rand(1).item() > 0.5:
            noise = np.random.normal(0, 2, img_np.shape).astype(np.int8)
            img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add compression artifacts occasionally
        if torch.rand(1).item() > 0.8:
            # Simulate JPEG compression
            import io
            img_pil = Image.fromarray(img_np)
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            img_pil = Image.open(buffer)
            img_np = np.array(img_pil)

        return Image.fromarray(img_np)


class SmartRandomInvert:
    """Intelligent inversion based on background detection"""

    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_gray = img.convert('L')
        img_np = np.array(img_gray)

        # Analyze background vs foreground
        background_intensity = np.mean([img_np[0, :], img_np[-1, :], img_np[:, 0], img_np[:, -1]])

        # Smart inversion decision
        if background_intensity > 127:  # White background
            if torch.rand(1).item() > 0.3:  # 70% chance to invert white bg
                return ImageOps.invert(img.convert('RGB'))
        else:  # Dark background
            if torch.rand(1).item() > 0.7:  # 30% chance to invert dark bg
                return ImageOps.invert(img.convert('RGB'))

        return img


class CoarseDropout:
    """Advanced dropout augmentation"""

    def __init__(self, p=0.3, max_holes=8, max_height=0.15, max_width=0.15):
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img)
        h, w, c = img_np.shape

        n_holes = random.randint(1, self.max_holes)

        for _ in range(n_holes):
            hole_h = int(h * random.uniform(0.02, self.max_height))
            hole_w = int(w * random.uniform(0.02, self.max_width))

            y1 = random.randint(0, h - hole_h)
            x1 = random.randint(0, w - hole_w)

            # Random fill color (background simulation)
            fill_color = random.choice([0, 255, random.randint(0, 255)])
            img_np[y1:y1+hole_h, x1:x1+hole_w, :] = fill_color

        return Image.fromarray(img_np)


class GridDistortion:
    """Grid-based distortion for tablet surface simulation"""

    def __init__(self, p=0.4, num_steps=5):
        self.p = p
        self.num_steps = num_steps

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Create distortion grid
        x_steps = np.linspace(0, w-1, self.num_steps)
        y_steps = np.linspace(0, h-1, self.num_steps)

        # Add random distortion to grid points
        x_distort = x_steps + np.random.normal(0, w*0.02, self.num_steps)
        y_distort = y_steps + np.random.normal(0, h*0.02, self.num_steps)

        # Create smooth distortion field
        from scipy.interpolate import griddata

        points = []
        values_x = []
        values_y = []

        for i, y in enumerate(y_steps):
            for j, x in enumerate(x_steps):
                points.append([y, x])
                values_x.append(x_distort[j] - x)
                values_y.append(y_distort[i] - y)

        # Apply distortion
        yi, xi = np.mgrid[0:h, 0:w]
        dx = griddata(points, values_x, (yi, xi), method='cubic', fill_value=0)
        dy = griddata(points, values_y, (yi, xi), method='cubic', fill_value=0)

        # Remap image
        map_x = (xi + dx).astype(np.float32)
        map_y = (yi + dy).astype(np.float32)

        if len(img_np.shape) == 3:
            distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(distorted)


class RandomSharpen:
    """Add sharpening to simulate digital pen precision"""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        # Unsharp mask
        img_np = np.array(img)
        gaussian = cv2.GaussianBlur(img_np, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)

        return Image.fromarray(sharpened)


class TabletPreprocessor:
    """Light preprocessing for validation to simulate tablet input"""

    def __call__(self, img):
        img_np = np.array(img.convert('L'))

        # Slight smoothing (tablet anti-aliasing)
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)

        # Ensure proper contrast
        img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=10)

        return Image.fromarray(img_np).convert('RGB')


class TabletCanvasProcessor:
    """Advanced processor for tablet canvas input"""

    def __call__(self, img):
        img_gray = img.convert('L')
        img_np = np.array(img_gray)

        # Auto-detect and normalize background
        background_intensity = np.percentile(img_np, 95)

        if background_intensity > 200:  # White background detected
            img_np = 255 - img_np  # Invert to match training data

        # Enhance stroke clarity
        img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=5)

        # Apply slight smoothing (tablet rendering)
        img_np = cv2.bilateralFilter(img_np, 5, 20, 20)

        # Ensure binary-like appearance
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(img_np).convert('RGB')


# Existing transforms with improvements
class RandomBlur:
    """Enhanced blur simulation"""

    def __init__(self, p=0.4, max_radius=2.0):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        radius = torch.rand(1).item() * self.max_radius
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class ElasticTransform:
    """Enhanced elastic deformation"""

    def __init__(self, p=0.5, alpha=15, sigma=3):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img_np = np.array(img)
        shape = img_np.shape[:2]

        # Generate smooth random displacement
        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create coordinate arrays
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply transformation
        if len(img_np.shape) == 3:
            distorted = np.empty_like(img_np)
            for i in range(img_np.shape[2]):
                distorted[:, :, i] = map_coordinates(
                    img_np[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            distorted = map_coordinates(
                img_np, indices, order=1, mode='reflect'
            ).reshape(shape)

        return Image.fromarray(distorted.astype(np.uint8))
