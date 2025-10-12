#!/usr/bin/env python3
"""
Quick Setup Script for Overnight Training
Validates environment and starts optimized training for maximum accuracy
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")

    required_packages = [
        'torch', 'torchvision', 'pillow', 'numpy', 'opencv-python',
        'scikit-learn', 'matplotlib', 'seaborn', 'scipy', 'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All dependencies installed")

    return True

def validate_data_structure():
    """Validate data directory structure"""
    print("\n📁 Validating data structure...")

    required_dirs = ['data/train', 'data/valid', 'mappings']
    missing_dirs = []

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path}")

    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False

    # Check if class mapping exists
    if os.path.exists('mappings/class_map.json'):
        print("✅ Class mapping found")
    else:
        print("❌ Class mapping not found - creating default...")
        create_default_class_mapping()

    return True

def create_default_class_mapping():
    """Create default class mapping if missing"""
    os.makedirs('mappings', exist_ok=True)

    # Create basic class mapping (you can customize this)
    class_mapping = {}
    for i in range(454):
        class_mapping[i] = f"Sinhala_Char_{i+1}"

    with open('mappings/class_map.json', 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    print("✅ Default class mapping created")

def check_system_resources():
    """Check system resources for optimal training"""
    print("\n🖥️  Checking system resources...")

    import torch

    # Check device availability
    if torch.backends.mps.is_available():
        print("✅ Apple MPS acceleration available")
        device = "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA GPU available")
        device = "cuda"
    else:
        print("⚠️  Only CPU available - training will be slow")
        device = "cpu"

    # Estimate memory usage
    import psutil
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"💾 Available RAM: {available_memory:.1f} GB")

    if available_memory < 8:
        print("⚠️  Low memory detected - consider reducing batch size")

    return device

def create_training_config(device):
    """Create optimized training configuration"""

    # Calculate optimal settings based on system
    if device == "mps":
        config = {
            "img_size": 112,
            "batch_size": 96,
            "num_workers": 6,
            "model_type": "mobilenet",
            "training_hours": 12.0,
            "dropout": 0.3
        }
    elif device == "cuda":
        config = {
            "img_size": 112,
            "batch_size": 128,
            "num_workers": 8,
            "model_type": "mobilenet",
            "training_hours": 12.0,
            "dropout": 0.3
        }
    else:  # CPU
        config = {
            "img_size": 96,
            "batch_size": 32,
            "num_workers": 2,
            "model_type": "mobilenet",
            "training_hours": 24.0,  # Longer for CPU
            "dropout": 0.4
        }

    return config

def main():
    print("🌙 SINHALA CHARACTER RECOGNITION - OVERNIGHT TRAINING SETUP")
    print("=" * 70)
    print(f"🕐 Setup started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Step 1: Check dependencies
        check_dependencies()

        # Step 2: Validate data structure
        if not validate_data_structure():
            print("\n❌ Data validation failed. Please ensure your data is properly structured.")
            return False

        # Step 3: Check system resources
        device = check_system_resources()

        # Step 4: Create optimal config
        config = create_training_config(device)

        print(f"\n🔧 OPTIMAL TRAINING CONFIGURATION:")
        print(f"📱 Device: {device}")
        print(f"🖼️  Image size: {config['img_size']}x{config['img_size']}")
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"👷 Workers: {config['num_workers']}")
        print(f"🏗️  Model: {config['model_type']}")
        print(f"⏰ Training hours: {config['training_hours']}")
        print(f"🎯 Target accuracy: 85-95%")

        # Step 5: Confirm and start training
        print(f"\n🚀 READY TO START OVERNIGHT TRAINING!")
        print("This will train your model for maximum accuracy with tablet optimization.")
        print("The training will automatically progress through 3 phases:")
        print("  Phase 1: Head-only training (fast learning)")
        print("  Phase 2: Partial fine-tuning (feature adaptation)")
        print("  Phase 3: Full fine-tuning (maximum accuracy)")

        response = input("\nStart overnight training? (y/N): ").strip().lower()

        if response == 'y':
            # Start training with optimal parameters
            cmd = [
                sys.executable, "overnight_train.py",
                "--img_size", str(config['img_size']),
                "--batch_size", str(config['batch_size']),
                "--num_workers", str(config['num_workers']),
                "--model_type", config['model_type'],
                "--hours", str(config['training_hours']),
                "--dropout", str(config['dropout'])
            ]

            print(f"\n🚀 Starting training with command:")
            print(f"python {' '.join(cmd[1:])}")
            print()

            # Execute training
            subprocess.run(cmd)

        else:
            print("\n⏸️  Training cancelled. You can start it manually with:")
            print(f"python overnight_train.py --hours {config['training_hours']} --batch_size {config['batch_size']}")

        return True

    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Setup completed successfully!")
    else:
        print(f"\n❌ Setup failed - please check the errors above")
