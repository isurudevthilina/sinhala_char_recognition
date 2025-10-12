#!/usr/bin/env python3
"""
Flask API Backend for Sinhala Character Recognition
Connects the web interface with the trained model
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import os
import sys
from datetime import datetime
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.infer.character_recognizer import SinhalaCharacterRecognizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global recognizer instance
recognizer = None

def initialize_recognizer():
    """Initialize the character recognizer"""
    global recognizer

    try:
        model_path = 'models/overnight_full_training_20251012_045934/best_model.pth'

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        print("🔄 Initializing Sinhala Character Recognizer...")
        recognizer = SinhalaCharacterRecognizer(
            model_path=model_path,
            device='mps' if __name__ == '__main__' else 'cpu',  # Use MPS when running directly
            confidence_threshold=0.01  # Lower threshold for more predictions
        )

        print("✅ Recognizer initialized successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize recognizer: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Serve the main interface"""
    try:
        # Get the correct path to the HTML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, 'tablet_interface.html')

        print(f"🔍 Looking for HTML file at: {html_path}")
        print(f"📂 File exists: {os.path.exists(html_path)}")

        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("✅ HTML file loaded successfully")
            return content

    except FileNotFoundError as e:
        print(f"❌ HTML file not found: {e}")
        # List files in the current directory for debugging
        current_dir = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(current_dir)
        print(f"📁 Files in {current_dir}: {files}")

        return """
        <h1>🎨 Sinhala Character Recognition Server</h1>
        <p><strong>Server is running successfully!</strong></p>
        <p>Interface file not found at expected location.</p>
        <h3>Available API endpoints:</h3>
        <ul>
            <li><strong>POST /api/recognize</strong> - Character recognition</li>
            <li><strong>GET /api/model-info</strong> - Model information</li>
            <li><strong>GET /api/health</strong> - Health check</li>
        </ul>
        <p>Model loaded with 454 Sinhala character classes and 90.48% accuracy!</p>
        """, 200
    except Exception as e:
        print(f"❌ Error serving HTML: {e}")
        return f"<h1>Server Error</h1><p>Error loading interface: {str(e)}</p>", 500

@app.route('/api/recognize', methods=['POST'])
def recognize_character():
    """
    API endpoint for character recognition
    Expects base64 encoded image data
    """
    try:
        if recognizer is None:
            return jsonify({
                'error': 'Model not initialized',
                'message': 'Character recognizer is not available'
            }), 500

        # Get image data from request
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'message': 'Please provide image data in base64 format'
            }), 400

        # Extract base64 image data
        image_data = data['image']

        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Get top_k parameter (default: 5)
        top_k = data.get('top_k', 5)
        top_k = min(max(1, top_k), 10)  # Limit between 1 and 10

        # Perform recognition
        predictions = recognizer.predict_character(image, top_k=top_k)

        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'num_classes': recognizer.num_classes,
                'model_accuracy': '90.48%',
                'top3_accuracy': '96.57%'
            }
        }

        return jsonify(response)

    except Exception as e:
        error_message = str(e)
        print(f"❌ Recognition error: {error_message}")
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': 'Recognition failed',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        if recognizer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500

        model_info = {
            'num_classes': recognizer.num_classes,
            'model_path': recognizer.model_path,
            'device': recognizer.device,
            'confidence_threshold': recognizer.confidence_threshold,
            'accuracy_metrics': {
                'overall_accuracy': '90.48%',
                'f1_score': '90.46%',
                'top3_accuracy': '96.57%',
                'top5_accuracy': '97.29%'
            },
            'capabilities': [
                'Real-time character recognition',
                'Graphic tablet pressure sensitivity support',
                '454 Sinhala character classes',
                'Confidence scoring',
                'Top-K predictions'
            ]
        }

        return jsonify(model_info)

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/batch-recognize', methods=['POST'])
def batch_recognize():
    """
    API endpoint for batch character recognition
    Expects array of base64 encoded images
    """
    try:
        if recognizer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500

        data = request.get_json()

        if not data or 'images' not in data:
            return jsonify({
                'error': 'No image data provided',
                'message': 'Please provide images array in base64 format'
            }), 400

        images_data = data['images']
        top_k = data.get('top_k', 5)

        if len(images_data) > 10:  # Limit batch size
            return jsonify({
                'error': 'Batch size too large',
                'message': 'Maximum 10 images per batch'
            }), 400

        # Process each image
        batch_results = []

        for i, image_data in enumerate(images_data):
            try:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                # Decode and process image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))

                predictions = recognizer.predict_character(image, top_k=top_k)

                batch_results.append({
                    'index': i,
                    'success': True,
                    'predictions': predictions
                })

            except Exception as e:
                batch_results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': batch_results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if recognizer is not None else 'unhealthy',
        'model_loaded': recognizer is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested API endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def main():
    """Main function to run the Flask app"""
    print("🚀 SINHALA CHARACTER RECOGNITION API SERVER")
    print("=" * 50)

    # Initialize recognizer
    if not initialize_recognizer():
        print("❌ Failed to start server: Model initialization failed")
        return

    print(f"✅ Server initialization complete!")
    print(f"🌐 Starting Flask server...")
    print(f"📱 Web interface will be available at: http://localhost:9000")
    print(f"🔗 API endpoints:")
    print(f"   • POST /api/recognize - Single character recognition")
    print(f"   • POST /api/batch-recognize - Batch recognition")
    print(f"   • GET /api/model-info - Model information")
    print(f"   • GET /api/health - Health check")
    print("=" * 50)

    try:
        # Run Flask app on port 9000 to avoid conflicts
        app.run(
            host='0.0.0.0',  # Accept connections from any IP
            port=9000,       # Changed to port 9000
            debug=True,
            use_reloader=False  # Disable reloader to avoid double initialization
        )
    except KeyboardInterrupt:
        print(f"\n⚠️ Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")

if __name__ == '__main__':
    main()
