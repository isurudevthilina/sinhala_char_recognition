import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
import time
from typing import List, Tuple, Optional, Dict
import cv2

from src.models.mobilenet import load_model_checkpoint
from src.dataset.transforms import get_inference_transforms


class TabletCanvasInference:
    """
    Real-time inference system optimized for tablet canvas input
    Handles paper-to-tablet domain adaptation with confidence scoring
    """

    def __init__(self, model_path: str, class_mapping_path: str,
                 device: str = 'auto', confidence_threshold: float = 0.7):
        """
        Initialize tablet canvas inference system

        Args:
            model_path: Path to trained model checkpoint
            class_mapping_path: Path to class mapping JSON
            device: Device to run inference on ('auto', 'mps', 'cuda', 'cpu')
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold

        # Load class mappings
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.num_classes = len(self.class_mapping)

        # Load model
        print(f"🔄 Loading model from {model_path}...")
        self.model = load_model_checkpoint(model_path, self.num_classes)
        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = get_inference_transforms(img_size=112)

        # Performance tracking
        self.inference_times = []

        print(f"✅ Tablet canvas inference system ready!")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Classes: {self.num_classes}")
        print(f"🔍 Confidence threshold: {confidence_threshold}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup optimal inference device"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_class_mapping(self, mapping_path: str) -> Dict:
        """Load class to character mapping"""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        # Convert keys to integers if they're strings
        if isinstance(list(mapping.keys())[0], str):
            mapping = {int(k): v for k, v in mapping.items()}

        return mapping

    def preprocess_canvas_image(self, canvas_image: Image.Image) -> torch.Tensor:
        """
        Advanced preprocessing for tablet canvas images
        Handles background detection, normalization, and domain adaptation
        """
        # Convert to numpy for processing
        img_np = np.array(canvas_image.convert('L'))

        # Auto-detect background type
        background_intensity = np.percentile(img_np, 95)

        # Smart background normalization
        if background_intensity > 200:  # White background
            # Invert to match training data (black background, white strokes)
            img_np = 255 - img_np

        # Enhance stroke clarity
        img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=5)

        # Apply slight smoothing (tablet anti-aliasing)
        img_np = cv2.bilateralFilter(img_np, 5, 20, 20)

        # Ensure binary-like appearance
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL and RGB
        processed_image = Image.fromarray(img_np).convert('RGB')

        return processed_image

    def predict_single(self, canvas_image: Image.Image,
                      return_confidence: bool = True,
                      return_top_k: int = 1) -> Dict:
        """
        Predict single character from tablet canvas image

        Args:
            canvas_image: PIL Image from tablet canvas
            return_confidence: Whether to return confidence scores
            return_top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions, characters, and confidence scores
        """
        start_time = time.time()

        # Preprocess canvas image
        processed_image = self.preprocess_canvas_image(canvas_image)

        # Apply transforms
        input_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get top-k predictions
            top_probs, top_classes = torch.topk(probabilities, return_top_k, dim=1)

            top_probs = top_probs.cpu().numpy()[0]
            top_classes = top_classes.cpu().numpy()[0]

        # Format results
        predictions = []
        for i in range(return_top_k):
            class_idx = top_classes[i]
            confidence = float(top_probs[i])

            # Get character from mapping
            character = self.class_mapping.get(class_idx, f"Class_{class_idx}")

            predictions.append({
                'class_id': int(class_idx),
                'character': character,
                'confidence': confidence,
                'above_threshold': confidence >= self.confidence_threshold
            })

        # Calculate inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return {
            'predictions': predictions,
            'best_prediction': predictions[0],
            'inference_time_ms': inference_time * 1000,
            'device': str(self.device)
        }

    def predict_batch(self, canvas_images: List[Image.Image]) -> List[Dict]:
        """
        Batch prediction for multiple canvas images
        More efficient for processing multiple characters
        """
        batch_start = time.time()

        # Preprocess all images
        processed_images = [self.preprocess_canvas_image(img) for img in canvas_images]

        # Apply transforms and create batch
        batch_tensors = []
        for img in processed_images:
            tensor = self.transform(img)
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors).to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get top prediction for each image
            top_probs, top_classes = torch.max(probabilities, dim=1)

            top_probs = top_probs.cpu().numpy()
            top_classes = top_classes.cpu().numpy()

        # Format batch results
        batch_results = []
        for i in range(len(canvas_images)):
            class_idx = top_classes[i]
            confidence = float(top_probs[i])
            character = self.class_mapping.get(class_idx, f"Class_{class_idx}")

            result = {
                'predictions': [{
                    'class_id': int(class_idx),
                    'character': character,
                    'confidence': confidence,
                    'above_threshold': confidence >= self.confidence_threshold
                }],
                'best_prediction': {
                    'class_id': int(class_idx),
                    'character': character,
                    'confidence': confidence,
                    'above_threshold': confidence >= self.confidence_threshold
                }
            }
            batch_results.append(result)

        batch_time = time.time() - batch_start

        print(f"📊 Batch inference: {len(canvas_images)} images in {batch_time*1000:.1f}ms "
              f"({batch_time/len(canvas_images)*1000:.1f}ms per image)")

        return batch_results

    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {"message": "No inference performed yet"}

        times_ms = [t * 1000 for t in self.inference_times]

        return {
            'total_inferences': len(self.inference_times),
            'average_time_ms': np.mean(times_ms),
            'median_time_ms': np.median(times_ms),
            'min_time_ms': np.min(times_ms),
            'max_time_ms': np.max(times_ms),
            'std_time_ms': np.std(times_ms),
            'fps': 1000 / np.mean(times_ms) if np.mean(times_ms) > 0 else 0.0
        }

    def validate_canvas_image(self, canvas_image: Image.Image) -> Dict:
        """
        Validate and analyze canvas image quality
        Helps debug tablet input issues
        """
        img_np = np.array(canvas_image.convert('L'))

        # Analyze image properties
        analysis = {
            'image_size': canvas_image.size,
            'format': canvas_image.format,
            'mode': canvas_image.mode,
            'background_intensity': float(np.percentile(img_np, 95)),
            'foreground_intensity': float(np.percentile(img_np, 5)),
            'contrast_ratio': float(np.std(img_np)),
            'has_content': np.std(img_np) > 10,  # Basic content detection
            'recommended_inversion': np.percentile(img_np, 95) > 200
        }

        # Quality assessment
        if analysis['contrast_ratio'] < 10:
            analysis['quality'] = 'poor'
            analysis['suggestion'] = 'Image has very low contrast. Check stylus pressure or canvas settings.'
        elif analysis['contrast_ratio'] < 30:
            analysis['quality'] = 'fair'
            analysis['suggestion'] = 'Image contrast could be improved for better recognition.'
        else:
            analysis['quality'] = 'good'
            analysis['suggestion'] = 'Image quality looks good for recognition.'

        return analysis


class TabletInferenceAPI:
    """
    REST API wrapper for tablet canvas inference
    Can be used with web interfaces
    """

    def __init__(self, model_path: str, class_mapping_path: str):
        self.inference_engine = TabletCanvasInference(model_path, class_mapping_path)

    def predict_from_base64(self, image_base64: str) -> Dict:
        """Predict from base64 encoded image (common in web apps)"""
        import base64
        from io import BytesIO

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        return self.inference_engine.predict_single(image)

    def predict_from_canvas_data(self, canvas_data: Dict) -> Dict:
        """
        Predict from HTML5 canvas data
        Expected format: {'width': int, 'height': int, 'imageData': base64_string}
        """
        # Extract canvas image
        canvas_image = self._canvas_data_to_image(canvas_data)

        return self.inference_engine.predict_single(canvas_image)

    def _canvas_data_to_image(self, canvas_data: Dict) -> Image.Image:
        """Convert HTML5 canvas data to PIL Image"""
        # This would be implemented based on your specific canvas format
        # For now, assume imageData is base64 encoded PNG
        import base64
        from io import BytesIO

        image_data = base64.b64decode(canvas_data['imageData'])
        return Image.open(BytesIO(image_data))


def create_inference_demo():
    """Create a simple demo for testing tablet canvas inference"""
    print("🎨 Tablet Canvas Inference Demo")
    print("=" * 50)

    # This would typically load your best trained model
    model_path = "models/best_checkpoint.pth"
    class_mapping_path = "mappings/class_map.json"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train a model first using overnight_train.py")
        return

    if not os.path.exists(class_mapping_path):
        print(f"❌ Class mapping not found at {class_mapping_path}")
        return

    # Initialize inference system
    inference_system = TabletCanvasInference(model_path, class_mapping_path)

    print("\n🎯 Inference system ready!")
    print("You can now integrate this with your tablet canvas interface.")
    print("\nExample usage:")
    print("result = inference_system.predict_single(canvas_image)")
    print("character = result['best_prediction']['character']")
    print("confidence = result['best_prediction']['confidence']")

    return inference_system


if __name__ == "__main__":
    create_inference_demo()
