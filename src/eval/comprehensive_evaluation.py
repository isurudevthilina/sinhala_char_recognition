#!/usr/bin/env python3
"""
Comprehensive Model Evaluation System
Generates detailed metrics, F1-scores, confusion matrices, and performance graphs
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd
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
from src.dataset.dataset import SinhalaCharDataset
from src.dataset.transforms import get_val_transforms
from src.utils.class_mapping import load_class_mappings


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with detailed metrics and visualizations"""

    def __init__(self, model_path, data_dir, device='mps', save_dir='evaluation_results'):
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Load class mappings
        self.class_to_idx, self.idx_to_class, self.unicode_map = load_class_mappings()
        self.num_classes = len(self.class_to_idx)

        print(f"🔬 Comprehensive Evaluator initialized")
        print(f"📊 Number of classes: {self.num_classes}")
        print(f"📱 Device: {device}")
        print(f"💾 Results will be saved to: {save_dir}")

    def load_model(self):
        """Load the trained model"""
        print("🔄 Loading trained model...")

        # Create model architecture
        model = get_model(
            num_classes=self.num_classes,
            pretrained=False,  # We're loading trained weights
            phase='full',
            model_type='mobilenet'
        )

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # Get training info
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.training_history = checkpoint.get('training_history', {})
        self.final_epoch = checkpoint.get('epoch', 'Unknown')

        print(f"✅ Model loaded successfully")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}%")
        print(f"📈 Final epoch: {self.final_epoch}")

        return model

    def create_test_loader(self, batch_size=32):
        """Create test data loader"""
        print("📊 Creating test dataset...")

        val_transforms = get_val_transforms(img_size=80)

        test_dataset = SinhalaCharDataset(
            root_dir=os.path.join(self.data_dir, 'test'),
            transform=val_transforms
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )

        print(f"📊 Test dataset size: {len(test_dataset):,} samples")
        print(f"📊 Number of batches: {len(test_loader):,}")

        return test_loader, test_dataset

    def evaluate_model(self, model, test_loader):
        """Comprehensive model evaluation"""
        print("🔬 Starting comprehensive evaluation...")

        model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if batch_idx % 100 == 0:
                    print(f"📊 Processed batch {batch_idx}/{len(test_loader)}")

        print("✅ Evaluation completed!")

        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

    def calculate_metrics(self, predictions, labels, probabilities):
        """Calculate comprehensive metrics"""
        print("📊 Calculating comprehensive metrics...")

        # Basic accuracy
        accuracy = accuracy_score(labels, predictions)

        # Precision, Recall, F1-score (macro and micro averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Top-K accuracy
        top3_accuracy = self.calculate_topk_accuracy(probabilities, labels, k=3)
        top5_accuracy = self.calculate_topk_accuracy(probabilities, labels, k=5)

        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        metrics = {
            'overall_accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'top3_accuracy': float(top3_accuracy),
            'top5_accuracy': float(top5_accuracy),
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'per_class_support': support_per_class.tolist(),
            'confusion_matrix': conf_matrix.tolist()
        }

        print(f"✅ Metrics calculated!")
        print(f"🎯 Overall Accuracy: {accuracy:.4f}")
        print(f"📈 F1-Score (Macro): {f1_macro:.4f}")
        print(f"📈 F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"🔝 Top-3 Accuracy: {top3_accuracy:.4f}")
        print(f"🔝 Top-5 Accuracy: {top5_accuracy:.4f}")

        return metrics

    def calculate_topk_accuracy(self, probabilities, labels, k=3):
        """Calculate top-k accuracy"""
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_predictions[i]:
                correct += 1
        return correct / len(labels)

    def create_detailed_report(self, metrics):
        """Create detailed classification report"""
        print("📋 Creating detailed classification report...")

        # Create per-class dataframe
        class_data = []
        for i in range(len(metrics['per_class_precision'])):
            class_name = self.idx_to_class.get(i, f'Class_{i}')
            unicode_char = self.unicode_map.get(class_name, '')

            class_data.append({
                'Class_ID': i,
                'Class_Name': class_name,
                'Unicode_Character': unicode_char,
                'Precision': metrics['per_class_precision'][i],
                'Recall': metrics['per_class_recall'][i],
                'F1_Score': metrics['per_class_f1'][i],
                'Support': metrics['per_class_support'][i]
            })

        df_classes = pd.DataFrame(class_data)

        # Save detailed report
        report_path = os.path.join(self.save_dir, 'detailed_classification_report.csv')
        df_classes.to_csv(report_path, index=False)

        # Find best and worst performing classes
        best_classes = df_classes.nlargest(10, 'F1_Score')
        worst_classes = df_classes.nsmallest(10, 'F1_Score')

        print(f"📊 Best performing classes (Top 10 F1-Score):")
        for idx, row in best_classes.iterrows():
            print(f"   {row['Unicode_Character']} ({row['Class_Name']}): F1={row['F1_Score']:.4f}")

        print(f"📊 Worst performing classes (Bottom 10 F1-Score):")
        for idx, row in worst_classes.iterrows():
            print(f"   {row['Unicode_Character']} ({row['Class_Name']}): F1={row['F1_Score']:.4f}")

        return df_classes, best_classes, worst_classes

    def plot_training_history(self):
        """Plot training history if available"""
        if not self.training_history:
            print("⚠️ No training history available")
            return

        print("📈 Creating training history plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')

        epochs = list(range(1, len(self.training_history['train_loss']) + 1))

        # Training & Validation Loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Training & Validation Accuracy
        axes[0, 1].plot(epochs, self.training_history['train_acc'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(epochs, self.training_history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate Analysis (if available)
        axes[1, 0].plot(epochs, self.training_history['val_acc'], color='green', linewidth=2)
        axes[1, 0].set_title('Validation Accuracy Progression')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Accuracy (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Accuracy Gap Analysis
        acc_gap = [val - train for val, train in zip(self.training_history['val_acc'], self.training_history['train_acc'])]
        axes[1, 1].plot(epochs, acc_gap, color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Validation - Training Accuracy Gap')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Training history plots saved")

    def plot_confusion_matrix(self, conf_matrix, top_n=50):
        """Plot confusion matrix for top N classes"""
        print(f"📊 Creating confusion matrix for top {top_n} classes...")

        # Get top N classes by support (most samples)
        class_support = np.sum(conf_matrix, axis=1)
        top_indices = np.argsort(class_support)[-top_n:]

        # Extract submatrix for top classes
        sub_matrix = conf_matrix[np.ix_(top_indices, top_indices)]

        # Normalize confusion matrix
        sub_matrix_norm = sub_matrix.astype('float') / sub_matrix.sum(axis=1)[:, np.newaxis]

        # Create labels for top classes
        labels = [self.idx_to_class.get(i, f'Class_{i}') for i in top_indices]

        # Plot
        plt.figure(figsize=(20, 16))
        sns.heatmap(sub_matrix_norm,
                   xticklabels=labels,
                   yticklabels=labels,
                   annot=False,  # Too many classes for annotations
                   cmap='Blues',
                   cbar_kws={'label': 'Normalized Frequency'})

        plt.title(f'Confusion Matrix - Top {top_n} Classes by Sample Count', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_top{top_n}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confusion matrix saved")

    def plot_performance_analysis(self, df_classes):
        """Plot detailed performance analysis"""
        print("📊 Creating performance analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

        # F1-Score distribution
        axes[0, 0].hist(df_classes['F1_Score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df_classes['F1_Score'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df_classes["F1_Score"].mean():.3f}')
        axes[0, 0].set_title('F1-Score Distribution Across Classes')
        axes[0, 0].set_xlabel('F1-Score')
        axes[0, 0].set_ylabel('Number of Classes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Precision vs Recall scatter
        axes[0, 1].scatter(df_classes['Recall'], df_classes['Precision'],
                          alpha=0.6, c=df_classes['F1_Score'], cmap='viridis', s=50)
        axes[0, 1].set_title('Precision vs Recall (colored by F1-Score)')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # Top 20 classes by F1-Score
        top_20 = df_classes.nlargest(20, 'F1_Score')
        axes[1, 0].bar(range(len(top_20)), top_20['F1_Score'], color='lightgreen')
        axes[1, 0].set_title('Top 20 Classes by F1-Score')
        axes[1, 0].set_xlabel('Class Rank')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom 20 classes by F1-Score
        bottom_20 = df_classes.nsmallest(20, 'F1_Score')
        axes[1, 1].bar(range(len(bottom_20)), bottom_20['F1_Score'], color='lightcoral')
        axes[1, 1].set_title('Bottom 20 Classes by F1-Score')
        axes[1, 1].set_xlabel('Class Rank')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Performance analysis plots saved")

    def save_results(self, metrics, df_classes):
        """Save all results to files"""
        print("💾 Saving evaluation results...")

        # Save metrics JSON
        metrics_path = os.path.join(self.save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save summary report
        summary_path = os.path.join(self.save_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("🔬 COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"📅 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🏆 Best Validation Accuracy: {self.best_val_acc:.4f}%\n")
            f.write(f"📈 Final Training Epoch: {self.final_epoch}\n\n")

            f.write("📊 OVERALL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}\n")
            f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}\n\n")

            f.write("📊 CLASS PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Classes: {len(df_classes)}\n")
            f.write(f"Mean F1-Score: {df_classes['F1_Score'].mean():.4f}\n")
            f.write(f"Std F1-Score: {df_classes['F1_Score'].std():.4f}\n")
            f.write(f"Min F1-Score: {df_classes['F1_Score'].min():.4f}\n")
            f.write(f"Max F1-Score: {df_classes['F1_Score'].max():.4f}\n\n")

            f.write("🏆 TOP 10 PERFORMING CLASSES:\n")
            f.write("-" * 30 + "\n")
            top_10 = df_classes.nlargest(10, 'F1_Score')
            for idx, row in top_10.iterrows():
                f.write(f"{row['Unicode_Character']} ({row['Class_Name']}): F1={row['F1_Score']:.4f}\n")

            f.write("\n📉 BOTTOM 10 PERFORMING CLASSES:\n")
            f.write("-" * 30 + "\n")
            bottom_10 = df_classes.nsmallest(10, 'F1_Score')
            for idx, row in bottom_10.iterrows():
                f.write(f"{row['Unicode_Character']} ({row['Class_Name']}): F1={row['F1_Score']:.4f}\n")

        print(f"✅ All results saved to: {self.save_dir}")
        print(f"📊 Evaluation metrics: {metrics_path}")
        print(f"📋 Detailed report: {os.path.join(self.save_dir, 'detailed_classification_report.csv')}")
        print(f"📄 Summary report: {summary_path}")

    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting comprehensive model evaluation...")
        print("=" * 50)

        # Load model
        model = self.load_model()

        # Create test loader
        test_loader, test_dataset = self.create_test_loader()

        # Evaluate model
        predictions, labels, probabilities = self.evaluate_model(model, test_loader)

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels, probabilities)

        # Create detailed report
        df_classes, best_classes, worst_classes = self.create_detailed_report(metrics)

        # Generate visualizations
        self.plot_training_history()
        self.plot_confusion_matrix(np.array(metrics['confusion_matrix']))
        self.plot_performance_analysis(df_classes)

        # Save all results
        self.save_results(metrics, df_classes)

        print("\n🎉 Comprehensive evaluation completed!")
        print("=" * 50)
        print(f"🏆 Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"📈 F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"🔝 Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        print(f"💾 All results saved to: {self.save_dir}")

        return metrics, df_classes


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model_path', type=str,
                       default='models/overnight_full_training_20251012_045934/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=device,
        save_dir=args.save_dir
    )

    # Run evaluation
    metrics, df_classes = evaluator.run_comprehensive_evaluation()

    return metrics, df_classes


if __name__ == '__main__':
    main()
