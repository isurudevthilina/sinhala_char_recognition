import os
import json
from collections import Counter


def check_dataset_structure(data_root='data'):
    """
    Verify dataset structure and count samples
    Checks train/val/test folders for all 454 classes
    """
    print("=" * 60)
    print("DATASET STRUCTURE VERIFICATION")
    print("=" * 60)

    splits = ['train', 'valid', 'test']
    results = {}

    for split in splits:
        split_path = os.path.join(data_root, split)

        if not os.path.exists(split_path):
            print(f"\nWARNING: {split_path} does not exist!")
            continue

        print(f"\n{split.upper()} SET:")
        print("-" * 40)

        total_samples = 0
        missing_folders = []
        class_counts = {}

        # Check each class folder (1-454)
        for class_id in range(1, 455):
            folder_name = str(class_id)
            folder_path = os.path.join(split_path, folder_name)

            if not os.path.exists(folder_path):
                missing_folders.append(folder_name)
                continue

            # Count images in folder
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            class_counts[folder_name] = count
            total_samples += count

        # Report statistics
        print(f"Total classes found: {len(class_counts)}/454")
        print(f"Total samples: {total_samples:,}")

        if class_counts:
            counts_list = list(class_counts.values())
            print(
                f"Samples per class - Min: {min(counts_list)}, Max: {max(counts_list)}, Avg: {sum(counts_list) / len(counts_list):.1f}")

        if missing_folders:
            print(f"\nMISSING FOLDERS ({len(missing_folders)}): {missing_folders[:10]}...")

        # Check for imbalanced classes
        if class_counts:
            counts_counter = Counter(class_counts.values())
            print(f"\nSample distribution:")
            for count, freq in sorted(counts_counter.items())[:5]:
                print(f"  {freq} classes have {count} samples")

        results[split] = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'missing_folders': missing_folders,
            'class_counts': class_counts
        }

    print("\n" + "=" * 60)
    return results


def verify_mapping_files():
    """Verify mapping JSON files exist and are valid"""
    print("\nMAPPING FILES VERIFICATION:")
    print("-" * 40)

    files = ['mappings/class_map.json', 'mappings/unicode_map.json']

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"MISSING: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"OK: {filepath} ({len(data)} entries)")
        except Exception as e:
            print(f"ERROR loading {filepath}: {e}")


def check_sample_images(data_root='data', num_samples=5):
    """Check if sample images can be loaded"""
    print("\nSAMPLE IMAGE LOADING TEST:")
    print("-" * 40)

    from PIL import Image

    train_path = os.path.join(data_root, 'train', '1')
    if not os.path.exists(train_path):
        print("Cannot find train/1/ folder for testing")
        return

    images = [f for f in os.listdir(train_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

    for img_name in images:
        img_path = os.path.join(train_path, img_name)
        try:
            img = Image.open(img_path)
            print(f"OK: {img_name} - Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            print(f"ERROR: {img_name} - {e}")


if __name__ == "__main__":
    # Run all checks
    results = check_dataset_structure()
    verify_mapping_files()
    check_sample_images()

    print("\n" + "=" * 60)
    print("DATASET CHECK COMPLETE")
    print("=" * 60)