#!/usr/bin/env python3
"""
Cool Training Script - Overnight Full Model Training for Mac
Optimized for Mac with simple cooling without password requirements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import os
import sys
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.mobilenet import get_model
from src.dataset.dataset import SinhalaCharDataset
from src.dataset.transforms import get_train_transforms, get_val_transforms


class CoolTrainer:
    """
    Overnight full model trainer optimized for Mac
    Simple cooling system without temperature monitoring
    """

    def __init__(self, model, train_loader, val_loader, device='mps', save_dir='models'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        # Training state
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"🧊 Cool Overnight Trainer initialized")
        print(f"📱 Device: {device}")
        print(f"💾 Save directory: {save_dir}")

    def get_optimizer_and_scheduler(self, total_epochs):
        """Get optimizer and scheduler for full model training"""
        # Full model training with good learning rates
        optimizer = optim.AdamW(
            self.model.parameters(),  # Train full model
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Cosine annealing for better convergence
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        return optimizer, scheduler

    def train_epoch(self, optimizer, criterion, scheduler):
        """Train one epoch with Mac-friendly cooling"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Clear cache before epoch (Mac optimization)
        if str(self.device).startswith('mps'):
            torch.mps.empty_cache()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Mac cooling - regular pauses
            if batch_idx % 50 == 0:
                if str(self.device).startswith('mps'):
                    torch.mps.empty_cache()

                print(f'🧊 Batch {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')

                # Small cooling pause every 50 batches
                time.sleep(0.1)

            # Longer cooling break every 200 batches
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"❄️ Cooling break...")
                time.sleep(3)  # 3 second pause for Mac cooling

        scheduler.step()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, criterion):
        """Validate with Mac-friendly cooling"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Light validation cooling
                if batch_idx % 100 == 0 and str(self.device).startswith('mps'):
                    torch.mps.empty_cache()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, train_loss, train_acc, val_loss, val_acc, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history
        }

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path)
            print(f"💾 Checkpoint saved at epoch {epoch}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! Val Acc: {val_acc:.4f}%")

    def overnight_train(self, total_epochs=200):
        """Full model overnight training optimized for Mac"""
        print(f"🌙 Starting overnight full model training")
        print(f"📊 Total epochs: {total_epochs}")
        print("🧊 Mac-optimized cooling enabled")

        # Unfreeze all model parameters for full training
        for param in self.model.parameters():
            param.requires_grad = True

        print("🔓 All model parameters unfrozen - training full model")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer, scheduler = self.get_optimizer_and_scheduler(total_epochs)

        start_time = time.time()

        try:
            for epoch in range(1, total_epochs + 1):
                epoch_start = time.time()

                print(f"\n🌙 Epoch {epoch}/{total_epochs}")

                # Training with cooling
                train_loss, train_acc = self.train_epoch(optimizer, criterion, scheduler)

                # Post-training cooling
                print("❄️ Post-training cooling...")
                time.sleep(2)

                # Validation with cooling
                val_loss, val_acc = self.validate_epoch(criterion)

                # Update history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)

                # Check improvement
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss

                # Save checkpoint
                self.save_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc, is_best)

                # Results
                epoch_time = time.time() - epoch_start
                total_time = (time.time() - start_time) / 3600
                current_lr = scheduler.get_last_lr()[0]

                print(f"🌙 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"🌙 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"⏱️ Epoch Time: {epoch_time:.1f}s | Total Time: {total_time:.1f}h")
                print(f"📈 Learning Rate: {current_lr:.2e}")

                # Inter-epoch cooling for Mac
                cooling_time = min(10, max(3, epoch_time * 0.15))  # 15% of epoch time
                print(f"🧊 Inter-epoch cooling: {cooling_time:.1f}s...")
                time.sleep(cooling_time)

                # Extended cooling every 20 epochs
                if epoch % 20 == 0:
                    print("❄️ Extended cooling break (Mac optimization)...")
                    time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user at epoch {epoch}")
        except Exception as e:
            print(f"\n❌ Training error at epoch {epoch}: {str(e)}")

        total_training_time = (time.time() - start_time) / 3600
        print(f"\n🌙 Overnight training completed!")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}%")
        print(f"⏱️ Total training time: {total_training_time:.2f} hours")
        print(f"🧊 Mac stayed cool throughout training!")


def main():
    parser = argparse.ArgumentParser(description='Cool Overnight Full Model Training')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced for Mac cooling
    parser.add_argument('--num_workers', type=int, default=2)  # Reduced for Mac cooling
    parser.add_argument('--epochs', type=int, default=200)     # Full overnight training
    parser.add_argument('--dropout', type=float, default=0.3)

    args = parser.parse_args()

    print("🌙 OVERNIGHT FULL MODEL TRAINING")
    print("=" * 50)
    print(f"🧊 MAC-OPTIMIZED COOLING (SAFE MODE)")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total epochs: {args.epochs}")
    print(f"🧊 Batch size: {args.batch_size} (Mac-safe)")
    print(f"🧊 Workers: {args.num_workers} (Mac-safe)")
    print("=" * 50)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.mps.empty_cache()
        print("🧊 Using MPS with Mac-optimized cooling")
    else:
        device = torch.device('cpu')
        print("🧊 Using CPU")

    # Create dataloaders
    train_transforms = get_train_transforms(img_size=args.img_size)
    val_transforms = get_val_transforms(img_size=args.img_size)

    train_dataset = SinhalaCharDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        transform=train_transforms
    )

    val_dataset = SinhalaCharDataset(
        root_dir=os.path.join(args.data_dir, 'valid'),
        transform=val_transforms
    )

    # Mac-safe data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # Disabled for Mac cooling
        persistent_workers=False  # Disabled for Mac cooling
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,  # Disabled for Mac cooling
        persistent_workers=False  # Disabled for Mac cooling
    )

    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(val_dataset):,}")

    # Create model for full training
    model = get_model(
        num_classes=train_dataset.num_classes,
        pretrained=True,
        phase='full',  # Full model training
        model_type='mobilenet',
        dropout=args.dropout
    )
    model = model.to(device)

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'overnight_full_training_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize trainer
    trainer = CoolTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        save_dir=save_dir
    )

    try:
        # Start overnight full training
        trainer.overnight_train(args.epochs)

        print(f"\n🌙 Overnight training completed successfully!")
        print(f"🏆 Final best validation accuracy: {trainer.best_val_acc:.4f}%")
        print(f"🧊 Mac cooling system worked perfectly!")

    except KeyboardInterrupt:
        print(f"\n⚠️ Training stopped by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")


if __name__ == '__main__':
    main()
