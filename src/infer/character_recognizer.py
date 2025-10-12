#!/usr/bin/env python3
"""
Sinhala Character Recognition Inference System
Real-time inference for graphic tablet input
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.mobilenet import get_model
from src.dataset.transforms import get_val_transforms
from src.utils.class_mapping import load_class_mappings


class SinhalaCharacterRecognizer:
    """
    Real-time Sinhala character recognition system
    Optimized for graphic tablet input
    """

    def __init__(self, model_path, device='mps', confidence_threshold=0.1):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load class mappings
        self.class_to_idx, self.idx_to_class, self.unicode_map = load_class_mappings()
        self.num_classes = len(self.class_to_idx)

        # Load model
        self.model = self._load_model()
        self.transforms = get_val_transforms(img_size=80)

        print(f"🎨 Sinhala Character Recognizer initialized")
        print(f"📊 Supporting {self.num_classes} character classes")
        print(f"📱 Device: {device}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")

    def _load_model(self):
        """Load the trained model"""
        print("🔄 Loading trained model...")

        # Create model architecture
        model = get_model(
            num_classes=self.num_classes,
            pretrained=False,
            phase='full',
            model_type='mobilenet'
        )

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # Get model info
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"✅ Model loaded successfully")
        print(f"🏆 Model validation accuracy: {best_val_acc:.2f}%")

        return model

    def preprocess_tablet_image(self, image, enhance_strokes=True):
        """
        Preprocess tablet drawing for recognition
        Handles various input formats and optimizes for character recognition
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image).convert('RGB')

        # Convert to numpy for preprocessing
        img_array = np.array(image)

        if enhance_strokes:
            img_array = self._enhance_tablet_strokes(img_array)

        # Convert back to PIL
        enhanced_image = Image.fromarray(img_array)

        # Apply model transforms
        tensor_image = self.transforms(enhanced_image)

        return tensor_image.unsqueeze(0)  # Add batch dimension

    def _enhance_tablet_strokes(self, image):
        """
        Enhance tablet strokes for better recognition
        Handles pressure variations, noise, and stroke thickness
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Detect if background is light or dark
        mean_intensity = np.mean(gray)
        is_light_bg = mean_intensity > 127

        # Enhance strokes
        if is_light_bg:
            # Dark strokes on light background
            # Threshold to make strokes more prominent
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Invert so strokes are white on black background (model expects this)
            enhanced = 255 - binary
        else:
            # Light strokes on dark background
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            enhanced = binary

        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

        # Ensure proper stroke thickness
        if np.sum(enhanced > 127) > 0:  # If there are stroke pixels
            # Slight dilation to ensure stroke connectivity
            kernel = np.ones((3,3), np.uint8)
            enhanced = cv2.dilate(enhanced, kernel, iterations=1)

        return enhanced

    def predict_character(self, image, top_k=5):
        """
        Predict Sinhala character from tablet drawing
        Returns top-k predictions with confidence scores
        """
        # Preprocess image
        input_tensor = self.preprocess_tablet_image(image)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes), dim=1)

            predictions = []
            for i in range(top_probs.size(1)):
                class_idx = top_indices[0, i].item()
                confidence = top_probs[0, i].item()

                # Only include predictions above threshold
                if confidence >= self.confidence_threshold:
                    class_name = self.idx_to_class.get(class_idx, f'Unknown_{class_idx}')
                    unicode_char = self.unicode_map.get(class_name, class_name)

                    predictions.append({
                        'character': unicode_char,
                        'class_name': class_name,
                        'class_id': class_idx,
                        'confidence': float(confidence),
                        'confidence_percent': float(confidence * 100)
                    })

        return predictions

    def predict_batch(self, images, top_k=5):
        """
        Predict multiple characters at once
        Useful for processing multiple drawings or segments
        """
        batch_predictions = []

        for image in images:
            predictions = self.predict_character(image, top_k=top_k)
            batch_predictions.append(predictions)

        return batch_predictions

    def get_character_info(self, class_name):
        """Get detailed information about a character"""
        class_idx = self.class_to_idx.get(class_name)
        unicode_char = self.unicode_map.get(class_name, class_name)

        return {
            'class_name': class_name,
            'class_id': class_idx,
            'unicode_character': unicode_char
        }

    def save_prediction_result(self, image, predictions, save_path=None):
        """Save prediction result with image and predictions"""
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'prediction_result_{timestamp}.json'

        # Convert image to base64 for storage (optional)
        result = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'model_path': self.model_path,
            'num_classes': self.num_classes
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 Prediction result saved to: {save_path}")

        return save_path


def create_demo_predictor():
    """Create a demo predictor instance"""
    model_path = 'models/overnight_full_training_20251012_045934/best_model.pth'

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Create recognizer
    recognizer = SinhalaCharacterRecognizer(
        model_path=model_path,
        device=device,
        confidence_threshold=0.05  # Lower threshold for demo
    )

    return recognizer


def test_with_sample_image(recognizer, test_image_path=None):
    """Test the recognizer with a sample image"""
    if test_image_path and os.path.exists(test_image_path):
        # Load test image
        test_image = Image.open(test_image_path)
        print(f"📸 Testing with image: {test_image_path}")
    else:
        # Create a simple test image (white square with black stroke)
        test_image = Image.new('RGB', (80, 80), 'white')
        # You can draw on this image for testing
        print(f"📸 Testing with generated image")

    # Get predictions
    predictions = recognizer.predict_character(test_image, top_k=5)

    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"=" * 40)

    if predictions:
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['character']} "
                  f"({pred['class_name']}) - "
                  f"{pred['confidence_percent']:.2f}%")
    else:
        print("No confident predictions found.")

    return predictions


def main():
    """Main demo function"""
    print("🎨 SINHALA CHARACTER RECOGNITION DEMO")
    print("=" * 50)

    # Create predictor
    recognizer = create_demo_predictor()

    if recognizer is None:
        print("❌ Failed to create recognizer")
        return

    # Test with sample
    print("\n🧪 Running test prediction...")
    test_predictions = test_with_sample_image(recognizer)

    print(f"\n✅ Inference system ready!")
    print(f"🎯 Model supports {recognizer.num_classes} Sinhala characters")
    print(f"📱 Optimized for graphic tablet input")
    print(f"⚡ Ready for real-time recognition")

    return recognizer


if __name__ == '__main__':
    recognizer = main()
