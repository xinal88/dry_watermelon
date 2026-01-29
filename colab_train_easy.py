"""
🚀 EASY COLAB TRAINING SCRIPT
Just copy your RAVDESS path and run!

Usage in Colab:
1. Mount Google Drive
2. Update RAVDESS_PATH below
3. Run this script!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add project root to path
sys.path.insert(0, '/content/multimodal-fer')

# ============================================================================
# 📝 CONFIGURATION - JUST CHANGE THIS!
# ============================================================================

CONFIG = {
    # 🔥 IMPORTANT: Update this to your Google Drive path
    "RAVDESS_PATH": "/content/drive/MyDrive/RAVDESS",
    
    # Save checkpoints to Google Drive (so you don't lose them!)
    "SAVE_DIR": "/content/drive/MyDrive/checkpoints/multimodal_fer",
    
    # Model configuration
    "model_type": "lightweight",  # "lightweight" or "full"
    
    # Training hyperparameters
    "batch_size": 8,
    "grad_accum_steps": 2,  # Effective batch size = 8 * 2 = 16
    "max_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    
    # Early stopping
    "early_stopping_patience": 15,
    "save_every": 5,  # Save checkpoint every N epochs
    
    # Data
    "modality": "speech",  # "speech" or "song"
    "num_workers": 2,  # Colab has multi-core
    "use_audio": True,  # Extract audio from videos
    
    # Logging
    "use_wandb": False,  # Set to True if you want WandB logging
    "wandb_project": "multimodal-fer-ravdess",
}

# ============================================================================
# 🔧 SETUP
# ============================================================================

def setup():
    """Setup environment and verify everything is ready."""
    print("="*70)
    print("🚀 MULTIMODAL FER TRAINING - EASY COLAB SETUP")
    print("="*70)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Check RAVDESS path
    ravdess_path = Path(CONFIG["RAVDESS_PATH"])
    print(f"\n📁 Checking RAVDESS dataset...")
    print(f"   Path: {ravdess_path}")
    
    if not ravdess_path.exists():
        print(f"\n❌ ERROR: RAVDESS path not found!")
        print(f"   Expected: {ravdess_path}")
        print(f"\n   Please update CONFIG['RAVDESS_PATH'] in this script")
        print(f"   Example: '/content/drive/MyDrive/RAVDESS'")
        sys.exit(1)
    
    # Count videos
    all_videos = list(ravdess_path.rglob("*.mp4"))
    print(f"✅ Found {len(all_videos)} total videos")
    
    if len(all_videos) < 100:
        print(f"\n⚠️  WARNING: Expected ~1440 videos, found {len(all_videos)}")
        print(f"   Dataset may be incomplete")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Create save directory
    save_dir = Path(CONFIG["SAVE_DIR"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Save directory: {save_dir}")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE - READY TO TRAIN!")
    print("="*70)
    
    return device


# ============================================================================
# 🏗️ CREATE MODEL
# ============================================================================

def create_model(model_type="lightweight"):
    """Create model with specified configuration."""
    from models import MultimodalFER, MultimodalFERConfig
    from models.audio_branch import AudioBranchConfig
    from models.visual_branch import VisualBranchConfig
    from models.fusion import LFM2FusionConfig
    from models.classifier import ClassifierConfig
    
    print("\n[1/4] Creating model...")
    
    if model_type == "lightweight":
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
                freeze_encoder=True,
                num_keep_tokens=64,
                temporal_depth=4,
                num_segments=8,
            ),
            fusion_config=LFM2FusionConfig(
                audio_dim=512,
                visual_dim=768,
                use_pretrained=False,
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
    else:  # full
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
                freeze_encoder=False,
                num_keep_tokens=64,
                temporal_depth=6,
                num_segments=8,
            ),
            fusion_config=LFM2FusionConfig(
                audio_dim=512,
                visual_dim=768,
                use_pretrained=True,
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
    
    model = MultimodalFER.from_config(config)
    model.print_summary()
    
    return model


# ============================================================================
# 📊 CREATE DATALOADERS
# ============================================================================

def create_dataloaders():
    """Create train/val/test dataloaders."""
    from data.simple_ravdess_dataset import create_simple_ravdess_dataloaders
    
    print("\n[2/4] Loading data...")
    
    try:
        train_loader, val_loader, test_loader = create_simple_ravdess_dataloaders(
            data_dir=CONFIG["RAVDESS_PATH"],
            modality=CONFIG["modality"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            use_audio=CONFIG["use_audio"],
        )
        
        print(f"✅ Train: {len(train_loader.dataset)} samples")
        print(f"✅ Val: {len(val_loader.dataset)} samples")
        print(f"✅ Test: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        print("\nPlease check:")
        print("  1. RAVDESS path is correct")
        print("  2. Videos are in .mp4 format")
        print("  3. Filenames follow RAVDESS format (XX-YY-ZZ-AA-BB-CC-DD.mp4)")
        sys.exit(1)


# ============================================================================
# 🎓 TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (audio, video, labels, _) in enumerate(pbar):
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        # Forward with mixed precision
        with autocast():
            outputs = model(audio, video)
            loss = criterion(outputs["logits"], labels)
            loss = loss / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * grad_accum_steps
        preds = outputs["probabilities"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
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
    
    for audio, video, labels, _ in tqdm(val_loader, desc="Validation"):
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
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
    """Save checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
        "config": CONFIG,
    }, save_path)
    
    print(f"  ✅ Saved: {save_path.name}")


# ============================================================================
# 🚀 MAIN TRAINING LOOP
# ============================================================================

def train():
    """Main training function."""
    # Setup
    device = setup()
    
    # Create model
    model = create_model(CONFIG["model_type"])
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Setup training
    print("\n[3/4] Setting up training...")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Differential learning rates
    param_groups = [
        {"params": model.audio_branch.parameters(), "lr": CONFIG["learning_rate"] * 0.1},
        {"params": model.visual_branch.parameters(), "lr": CONFIG["learning_rate"] * 0.1},
        {"params": model.fusion.parameters(), "lr": CONFIG["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["learning_rate"]},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    # WandB (optional)
    if CONFIG["use_wandb"]:
        try:
            import wandb
            wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
            print("✅ WandB initialized")
        except:
            print("⚠️  WandB not available, skipping")
            CONFIG["use_wandb"] = False
    
    print("✅ Training setup complete")
    
    # Training loop
    print("\n[4/4] Training...")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(CONFIG["max_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['max_epochs']}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            CONFIG["grad_accum_steps"]
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        # WandB logging
        if CONFIG["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc,
                Path(CONFIG["SAVE_DIR"]) / "best_model.pth"
            )
            print(f"  🎉 New best! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Periodic checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc,
                Path(CONFIG["SAVE_DIR"]) / f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\n⚠️  Early stopping (patience={CONFIG['early_stopping_patience']})")
            break
        
        scheduler.step()
    
    # Save history
    history_path = Path(CONFIG["SAVE_DIR"]) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Test
    print("\n" + "="*70)
    print("Testing on test set...")
    print("-"*70)
    
    checkpoint = torch.load(Path(CONFIG["SAVE_DIR"]) / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    if CONFIG["use_wandb"]:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb.finish()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Total Epochs: {epoch + 1}")
    print(f"\nCheckpoints saved in: {CONFIG['SAVE_DIR']}")
    print("="*70)


# ============================================================================
# 🎯 RUN
# ============================================================================

if __name__ == "__main__":
    train()
