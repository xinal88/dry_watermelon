"""
Complete Training Script for Google Colab Pro
Optimized for RAVDESS dataset with Multimodal FER

Features:
- Mixed precision training (FP16)
- Gradient accumulation
- Checkpointing
- WandB logging
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER, MultimodalFERConfig
from models.audio_branch import AudioBranchConfig
from models.visual_branch import VisualBranchConfig
from models.fusion import LFM2FusionConfig
from models.classifier import ClassifierConfig


def create_model(config_type="lightweight"):
    """Create model with different configurations."""
    
    if config_type == "lightweight":
        # Lightweight config for fast training
        config = MultimodalFERConfig(
            audio_config=AudioBranchConfig(
                pretrained_model=None,
                feature_dim=512,
                use_nemo=False,
                num_layers=4,
                num_segments=8,
            ),
            visual_config=VisualBranchConfig(
                pretrained_model="google/siglip-base-patch16-224",
                feature_dim=768,
                freeze_encoder=True,  # Freeze SigLIP
                num_keep_tokens=64,
                temporal_depth=4,
                num_segments=8,
            ),
            fusion_config=LFM2FusionConfig(
                audio_dim=512,
                visual_dim=768,
                use_pretrained=False,  # Custom LFM2
                num_layers=4,
                output_dim=512,
            ),
            classifier_config=ClassifierConfig(
                input_dim=512,
                hidden_dims=[512, 256],
                num_classes=8,
                dropout=0.1,
            ),
            num_classes=8,
            num_segments=8,
        )
    
    elif config_type == "full":
        # Full config with pretrained LFM2
        config = MultimodalFERConfig(
            audio_config=AudioBranchConfig(
                pretrained_model=None,
                feature_dim=512,
                use_nemo=False,
                num_layers=6,
                num_segments=8,
            ),
            visual_config=VisualBranchConfig(
                pretrained_model="google/siglip-base-patch16-224",
                feature_dim=768,
                freeze_encoder=False,  # Finetune SigLIP
                num_keep_tokens=64,
                temporal_depth=6,
                num_segments=8,
            ),
            fusion_config=LFM2FusionConfig(
                audio_dim=512,
                visual_dim=768,
                use_pretrained=True,  # Pretrained LFM2
                num_layers=6,
                output_dim=512,
            ),
            classifier_config=ClassifierConfig(
                input_dim=512,
                hidden_dims=[512, 256],
                num_classes=8,
                dropout=0.1,
            ),
            num_classes=8,
            num_segments=8,
        )
    
    else:
        raise ValueError(f"Unknown config_type: {config_type}")
    
    model = MultimodalFER.from_config(config)
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps=1):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        
        # Forward with mixed precision
        with autocast():
            outputs = model(audio, video)
            loss = criterion(outputs["logits"], labels)
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * grad_accum_steps
        preds = outputs["probabilities"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "acc": f"{100*correct/total:.2f}%"
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(audio, video)
        loss = criterion(outputs["logits"], labels)
        
        total_loss += loss.item()
        preds = outputs["probabilities"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, save_path):
    """Save model checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
    }, save_path)
    
    print(f"  ✅ Saved checkpoint: {save_path}")


def train_colab(
    data_dir: str,
    save_dir: str = "checkpoints",
    config_type: str = "lightweight",
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    max_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 15,
    save_every: int = 5,
    use_wandb: bool = False,
    wandb_project: str = "multimodal-fer",
):
    """
    Complete training function for Google Colab.
    
    Args:
        data_dir: Path to RAVDESS dataset
        save_dir: Directory to save checkpoints
        config_type: "lightweight" or "full"
        batch_size: Batch size per GPU
        grad_accum_steps: Gradient accumulation steps
        max_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        early_stopping_patience: Patience for early stopping
        save_every: Save checkpoint every N epochs
        use_wandb: Use Weights & Biases logging
        wandb_project: WandB project name
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print("MULTIMODAL FER TRAINING - GOOGLE COLAB")
    print("="*70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    print("\n[1/6] Creating model...")
    model = create_model(config_type=config_type)
    model = model.to(device)
    model.print_summary()
    
    # Create dataloaders
    print("\n[2/6] Loading data...")
    try:
        from data.ravdess_dataset import create_ravdess_dataloaders
        
        train_loader, val_loader, test_loader = create_ravdess_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2,  # Colab has multi-core
            modality="both",
            use_audio=True,
        )
        
        print(f"✅ Train: {len(train_loader.dataset)} samples")
        print(f"✅ Val: {len(val_loader.dataset)} samples")
        print(f"✅ Test: {len(test_loader.dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nPlease ensure:")
        print("  1. RAVDESS dataset is in the correct location")
        print("  2. Dataset structure is correct (Actor_XX folders)")
        print("  3. Video files are .mp4 format")
        return None
    
    # Setup training
    print("\n[3/6] Setting up training...")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with differential learning rates
    param_groups = [
        {"params": model.audio_branch.parameters(), "lr": lr * 0.1},
        {"params": model.visual_branch.parameters(), "lr": lr * 0.1},
        {"params": model.fusion.parameters(), "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # WandB
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "config_type": config_type,
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "effective_batch_size": batch_size * grad_accum_steps,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_epochs": max_epochs,
                }
            )
            print("✅ WandB initialized")
        except ImportError:
            print("⚠️ WandB not installed, skipping logging")
            use_wandb = False
    
    print(f"✅ Optimizer: AdamW (lr={lr})")
    print(f"✅ Scheduler: CosineAnnealingWarmRestarts")
    print(f"✅ Loss: CrossEntropyLoss (label_smoothing=0.1)")
    print(f"✅ Mixed precision: FP16")
    print(f"✅ Gradient accumulation: {grad_accum_steps} steps")
    print(f"✅ Effective batch size: {batch_size * grad_accum_steps}")
    
    # Training loop
    print("\n[4/6] Training...")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })
        
        # WandB logging
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc,
                Path(save_dir) / "best_model.pth"
            )
            print(f"  🎉 New best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc,
                Path(save_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⚠️ Early stopping triggered (patience={early_stopping_patience})")
            print(f"   Best val acc: {best_val_acc:.2f}%")
            break
        
        # Step scheduler
        scheduler.step()
    
    # Save training history
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\n✅ Saved training history: {history_path}")
    
    print("\n[5/6] Training complete!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total epochs: {epoch + 1}")
    
    # Test
    print("\n[6/6] Testing on test set...")
    print("-"*70)
    
    # Load best model
    checkpoint = torch.load(Path(save_dir) / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        wandb.finish()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nCheckpoints saved in: {save_dir}")
    print(f"Best model: {Path(save_dir) / 'best_model.pth'}")
    print(f"Val Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal FER on Google Colab")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to RAVDESS dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    # Model
    parser.add_argument("--config_type", type=str, default="lightweight",
                        choices=["lightweight", "full"],
                        help="Model configuration")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="multimodal-fer",
                        help="WandB project name")
    
    args = parser.parse_args()
    
    # Train
    train_colab(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        config_type=args.config_type,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        save_every=args.save_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
