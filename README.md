# 🎨 Sinhala Character Recognition System

A state-of-the-art AI-powered system for recognizing handwritten Sinhala characters with graphic tablet support. This project combines deep learning with a user-friendly web interface to provide real-time character recognition for the Sinhala script.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Project Overview

This system recognizes **454 different Sinhala characters** with **90.48% accuracy** using a custom-trained MobileNetV3-Large model. It features a responsive web interface optimized for graphic tablets, making it perfect for digital handwriting recognition and educational applications.

### ✨ Key Features

- 🎯 **High Accuracy**: 90.48% overall accuracy, 96.57% top-3 accuracy
- 📱 **454 Character Classes**: Complete Sinhala script support
- 🖊️ **Tablet Optimized**: Pressure-sensitive drawing with graphic tablet support
- ⚡ **Real-time Recognition**: Fast inference (~100ms per character)
- 🌐 **Web Interface**: Beautiful, responsive UI for easy interaction
- 🔥 **MPS Acceleration**: Optimized for Apple Silicon Macs
- 📊 **Comprehensive Analytics**: Detailed performance metrics and evaluation

## 🏗️ Architecture

### Model Architecture
- **Base Model**: MobileNetV3-Large (ImageNet pretrained)
- **Custom Classifier**: 4-layer tablet-optimized head
- **Attention Mechanism**: Squeeze-and-Excitation blocks
- **Input Resolution**: 80×80 grayscale images
- **Parameters**: 9.56M total parameters
- **Training Strategy**: Progressive training (head → partial → full)

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │───▶│   Flask API     │───▶│   AI Model      │
│   (HTML/CSS/JS) │    │   (REST API)    │    │   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Tablet Drawing  │    │ Image Processing│    │ Character       │
│ Canvas          │    │ & Preprocessing │    │ Classification  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```
## Dataset

This project uses the **Sinhala Letter Dataset** from Kaggle.
- **Dataset**: [Sinhala Letter 454](https://www.kaggle.com/datasets/sathiralamal/sinhala-letter-454)
- **Creator**: Sathira Lamal
- **Source**: Kaggle

### Dataset Details
The dataset contains 454 classes of Sinhala letters and characters, providing comprehensive coverage for Sinhala handwriting recognition tasks.

### License
Please refer to the [dataset page](https://www.kaggle.com/datasets/sathiralamal/sinhala-letter-454) for licensing information.

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Main programming language
- **PyTorch 2.0+**: Deep learning framework
- **torchvision**: Computer vision utilities and pretrained models
- **MobileNetV3-Large**: Efficient CNN architecture

### Web Technologies
- **Flask 2.0+**: Web framework for REST API
- **Flask-CORS**: Cross-origin resource sharing
- **HTML5 Canvas**: Interactive drawing interface
- **JavaScript ES6**: Client-side interactivity
- **CSS3**: Modern responsive styling

### Data Processing
- **PIL (Pillow)**: Image processing and manipulation
- **OpenCV (cv2)**: Advanced image preprocessing
- **NumPy**: Numerical computations
- **pandas**: Data analysis and manipulation

### Machine Learning & Analytics
- **scikit-learn**: Model evaluation metrics
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization
- **F1-score, Precision, Recall**: Performance metrics

### Development Tools
- **argparse**: Command-line argument parsing
- **datetime**: Time tracking and logging
- **json**: Configuration and data storage
- **base64**: Image encoding/decoding for web API
- **warnings**: Error handling and debugging

### Hardware Optimization
- **MPS (Metal Performance Shaders)**: Apple Silicon acceleration
- **CUDA Support**: NVIDIA GPU compatibility
- **CPU Fallback**: Universal compatibility

## 📁 Project Structure

```
sinhala_char_recognition/
├── 📄 README.md                     # Project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 analyze_class_identification.py # Class analysis script
├── 📄 setup_training.py            # Training setup utilities
│
├── 📂 data/                        # Dataset directory
│   ├── 📂 train/                  # Training data (454 classes)
│   ├── 📂 valid/                  # Validation data
│   └── 📂 test/                   # Test data
│
├── 📂 src/                         # Source code
│   ├── 📂 models/                 # Model architectures
│   │   └── 📄 mobilenet.py        # MobileNetV3 implementation
│   ├── 📂 dataset/                # Dataset handling
│   │   ├── 📄 dataset.py          # Dataset class
│   │   └── 📄 transforms.py       # Data augmentation
│   ├── 📂 train/                  # Training scripts
│   │   └── 📄 cool_trainer.py     # Mac-optimized trainer
│   ├── 📂 eval/                   # Evaluation tools
│   │   └── 📄 comprehensive_evaluation.py
│   ├── 📂 infer/                  # Inference system
│   │   └── 📄 character_recognizer.py
│   ├── 📂 web/                    # Web interface
│   │   ├── 📄 api_server.py       # Flask REST API
│   │   └── 📄 tablet_interface.html # Web UI
│   └── 📂 utils/                  # Utilities
│       └── 📄 class_mapping.py    # Class management
│
├── 📂 models/                      # Trained models
│   └── 📂 overnight_full_training_20251012_045934/
│       ├── 📄 best_model.pth      # Best trained model
│       └── 📄 latest_checkpoint.pth
│
├── 📂 mappings/                    # Character mappings
│   ├── 📄 class_map.json         # Class to index mapping
│   └── 📄 unicode_map.json       # Unicode character mapping
│
├── 📂 evaluation_results/          # Model evaluation
│   ├── 📄 evaluation_metrics.json
│   ├── 📄 evaluation_summary.txt
│   ├── 📊 confusion_matrix_top50.png
│   ├── 📊 performance_analysis.png
│   └── 📊 training_history.png
│
└── 📂 docs/                       # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- macOS with Apple Silicon (for MPS acceleration) or any system with CPU/GPU
- Graphic tablet (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sinhala_char_recognition
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install additional packages**
```bash
pip install flask flask-cors opencv-python scikit-learn matplotlib seaborn
```

### Usage

#### 1. Start the Web Application
```bash
python src/web/api_server.py
```
Then open your browser and go to: **http://localhost:9000**

#### 2. Use the Drawing Interface
- Draw Sinhala characters on the canvas using your graphic tablet or mouse
- Adjust brush size and opacity as needed
- Click "🔍 Recognize Character" to get AI predictions
- View top 5 predictions with confidence scores

#### 3. API Usage
```python
import requests
import base64

# Load your image
with open("character_image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post("http://localhost:9000/api/recognize", 
                        json={"image": image_data, "top_k": 5})

predictions = response.json()["predictions"]
```

## 🎯 Model Performance

### Overall Metrics
- **Accuracy**: 90.48%
- **F1-Score (Macro)**: 90.46%
- **F1-Score (Weighted)**: 90.46%
- **Top-3 Accuracy**: 96.57%
- **Top-5 Accuracy**: 97.29%

### Performance Distribution
- **Excellent (F1 ≥ 0.9)**: 387 classes (85.2%)
- **Good (0.7 ≤ F1 < 0.9)**: 45 classes (9.9%)
- **Average (0.5 ≤ F1 < 0.7)**: 18 classes (4.0%)
- **Poor (0 < F1 < 0.5)**: 4 classes (0.9%)

### Training Details
- **Training Time**: 10.4 hours (200 epochs)
- **Dataset Size**: 87,141 training samples
- **Validation Size**: ~21,785 samples
- **Test Size**: 10,896 samples (24 per class)
- **Batch Size**: 32 (Mac-optimized)
- **Learning Rate**: Cosine annealing (1e-3 → 1e-6)

## 🔧 Training Your Own Model

### 1. Prepare Your Dataset
```bash
# Organize your data in this structure:
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 2. Run Training
```bash
# Full overnight training (recommended)
python src/train/cool_trainer.py --epochs 200 --batch_size 32

# Quick training for testing
python src/train/cool_trainer.py --epochs 50 --batch_size 16
```

### 3. Evaluate Model
```bash
python src/eval/comprehensive_evaluation.py --model_path models/your_model/best_model.pth
```

## 📊 Model Evaluation

The system includes comprehensive evaluation tools:

```bash
# Run complete evaluation
python src/eval/comprehensive_evaluation.py

# Analyze class performance
python analyze_class_identification.py
```

**Evaluation Outputs:**
- Confusion matrix visualization
- Per-class performance metrics
- Training history plots
- F1-score distributions
- Detailed classification reports

## 🎨 Web Interface Features

### Drawing Tools
- **Pressure-sensitive drawing** for natural tablet input
- **Adjustable brush size** (1-20 pixels)
- **Variable opacity** (10-100%)
- **Undo/Redo functionality**
- **Clear canvas** option

### Recognition Features
- **Real-time AI inference** (~100ms response time)
- **Top-K predictions** (configurable, default: 5)
- **Confidence scoring** for each prediction
- **Session statistics** tracking
- **Unicode character display**

### Interface Design
- **Responsive design** for tablets and desktops
- **Modern gradient styling**
- **Intuitive controls** and tooltips
- **Real-time feedback** and status updates
- **Professional layout** with model statistics

## 🔌 API Endpoints

### Character Recognition
```http
POST /api/recognize
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "top_k": 5
}
```

### Batch Processing
```http
POST /api/batch-recognize
Content-Type: application/json

{
  "images": ["image1_base64", "image2_base64", ...],
  "top_k": 5
}
```

### Model Information
```http
GET /api/model-info
```

### Health Check
```http
GET /api/health
```

## 🚀 Deployment

### Local Development
```bash
# Development server
python src/web/api_server.py
```

### Production Deployment
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn --bind 0.0.0.0:9000 --workers 4 src.web.api_server:app

# Using Docker
docker build -t sinhala-recognition .
docker run -p 9000:9000 sinhala-recognition
```

### Environment Variables
```bash
export MODEL_PATH=models/your_model/best_model.pth
export DEVICE=mps  # or cuda, cpu
export PORT=9000
export DEBUG=False
```

## 🧪 Testing

### Unit Tests
```bash
# Test model inference
python -c "from src.infer.character_recognizer import *; test_inference()"

# Test API endpoints
python -c "import requests; print(requests.get('http://localhost:9000/api/health').json())"
```

### Performance Testing
```bash
# Batch inference timing
python scripts/benchmark_inference.py

# Memory usage analysis
python scripts/memory_profiling.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update README for significant changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MobileNetV3** architecture from Google Research
- **PyTorch** team for the excellent deep learning framework
- **Sinhala language community** for inspiration and support
- **Open source contributors** for various utilities and tools

## 📞 Support

For questions, issues, or contributions:

- 📧 **Email**: isurudev2004@gmail.com


**Built with ❤️ for the Sinhala language community**

*This project demonstrates the power of modern AI in preserving and digitizing cultural scripts and languages.*

