#!/usr/bin/env python3
"""
Class Mapping Utilities
Load and manage class mappings for Sinhala character recognition
"""

import json
import os


def load_class_mappings(mappings_dir='mappings'):
    """
    Load class mappings from JSON files
    Returns class_to_idx, idx_to_class, and unicode_map
    """

    # Paths to mapping files
    class_map_path = os.path.join(mappings_dir, 'class_map.json')
    unicode_map_path = os.path.join(mappings_dir, 'unicode_map.json')

    # Load class mapping
    try:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            raw_class_map = json.load(f)

        # Handle the nested structure: "0": {"folder_name": "1", "character": "අ"}
        class_to_idx = {}  # folder_name -> index
        idx_to_class = {}  # index -> folder_name
        unicode_map = {}   # folder_name -> character

        for idx_str, class_info in raw_class_map.items():
            idx = int(idx_str)  # Convert string index to int
            folder_name = class_info['folder_name']
            character = class_info['character']

            class_to_idx[folder_name] = idx
            idx_to_class[idx] = folder_name
            unicode_map[folder_name] = character

        print(f"✅ Loaded {len(class_to_idx)} class mappings from {class_map_path}")
        print(f"✅ Loaded {len(unicode_map)} unicode mappings")

    except FileNotFoundError:
        print(f"⚠️ Class mapping file not found: {class_map_path}")
        print("🔄 Generating class mapping from dataset...")
        class_to_idx, idx_to_class = generate_class_mapping_from_dataset()
        unicode_map = create_basic_unicode_mapping(class_to_idx)
    except Exception as e:
        print(f"⚠️ Error loading class mapping: {e}")
        print("🔄 Generating class mapping from dataset...")
        class_to_idx, idx_to_class = generate_class_mapping_from_dataset()
        unicode_map = create_basic_unicode_mapping(class_to_idx)

    return class_to_idx, idx_to_class, unicode_map


def generate_class_mapping_from_dataset(data_dir='data/train'):
    """
    Generate class mapping by scanning the training dataset directory
    """
    import os

    print(f"🔍 Scanning dataset directory: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Get all class directories
    class_names = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)

    # Sort to ensure consistent ordering
    class_names.sort()

    # Create mappings
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    print(f"📊 Generated mappings for {len(class_names)} classes")

    # Save the generated mapping
    os.makedirs('mappings', exist_ok=True)

    class_map_path = os.path.join('mappings', 'class_map.json')
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved class mapping to: {class_map_path}")

    return class_to_idx, idx_to_class


def create_basic_unicode_mapping(class_to_idx):
    """
    Create a basic unicode mapping
    In a real scenario, this would map class names to actual Sinhala unicode characters
    """
    unicode_map = {}

    for class_name in class_to_idx.keys():
        # For now, we'll use the class name as the display character
        # In a real implementation, you'd map these to actual Sinhala unicode characters
        unicode_map[class_name] = class_name

    # Save the basic mapping
    os.makedirs('mappings', exist_ok=True)

    unicode_map_path = os.path.join('mappings', 'unicode_map.json')
    with open(unicode_map_path, 'w', encoding='utf-8') as f:
        json.dump(unicode_map, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved unicode mapping to: {unicode_map_path}")

    return unicode_map


def get_class_name_from_idx(idx, idx_to_class):
    """Get class name from class index"""
    return idx_to_class.get(idx, f'Unknown_Class_{idx}')


def get_unicode_char_from_class(class_name, unicode_map):
    """Get unicode character from class name"""
    return unicode_map.get(class_name, class_name)


def print_class_statistics(class_to_idx, data_dir='data'):
    """Print statistics about class distribution"""
    print("\n📊 CLASS DISTRIBUTION STATISTICS:")
    print("=" * 40)

    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        print(f"\n📁 {split.upper()} SET:")
        total_samples = 0

        for class_name in sorted(class_to_idx.keys()):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_samples += count

                if count > 0:  # Only print classes with samples
                    print(f"  {class_name}: {count:,} samples")

        print(f"  📊 Total {split} samples: {total_samples:,}")


if __name__ == '__main__':
    # Test the class mapping functions
    print("🧪 Testing class mapping utilities...")

    class_to_idx, idx_to_class, unicode_map = load_class_mappings()

    print(f"\n📊 Class mapping summary:")
    print(f"Total classes: {len(class_to_idx)}")
    print(f"Sample mappings:")
    for i, (class_name, idx) in enumerate(list(class_to_idx.items())[:5]):
        unicode_char = unicode_map.get(class_name, class_name)
        print(f"  {class_name} → {idx} (Unicode: {unicode_char})")

    # Print class statistics
    print_class_statistics(class_to_idx)
