"""
Resume training from checkpoint

Usage:
    python scripts/resume_training.py checkpoints/half_dataset_rtx3050/checkpoint_epoch_20.pt
"""

import sys
import argparse
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json

# Import the training script
from scripts.train_half_dataset import (
    create_half_dataloaders,
    train_epoch,
    validate,
    EmotionLoss,
    EmotionMetrics,
    GradScaler,
    tqdm,
    time,
    np
)

# Import model configs
from models import AudioBranchConfig, VisualBranchConfig, LFM2FusionConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs (default: from config)")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print("="*70)
    print("RESUME TRAINING")
    print("="*70)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]
    start_epoch = checkpoint["epoch"] + 1
    best_uar = checkpoint.get("best_uar", 0.0)
    
    if args.epochs:
        config["num_epochs"] = args.epochs
    
    print(f"\nResuming from epoch {start_epoch}/{config['num_epochs']}")
    print(f"Best UAR so far: {best_uar:.4f}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(config["save_dir"])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_half_dataloaders(
        data_dir=config["data_dir"],
        modality=config["modality"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    
    # Create model
    print("Creating model...")
    from models.classifier import ClassifierConfig
    
    # Need to recreate with same config
    audio_config = AudioBranchConfig(feature_dim=256, num_layers=6, num_segments=8)
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=False,
        feature_dim=256,
        temporal_depth=3
    )
    fusion_config = LFM2FusionConfig(
        use_pretrained=False,
        num_layers=3,
        hidden_dim=512,
        audio_dim=256,
        visual_dim=256,
        output_dim=256,
    )
    classifier_config = ClassifierConfig(
        input_dim=256,
        hidden_dims=[256, 128],
        num_classes=8,
        dropout=0.1,
    )
    
    from models import MultimodalFER
    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        classifier_config=classifier_config,
        num_classes=8,
        num_segments=8,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Optimizer
    criterion = EmotionLoss(num_classes=8, label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler()
    metrics_calculator = EmotionMetrics(num_classes=8)
    
    # Load history
    history_path = save_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_uar": [],
            "epoch_time": [],
        }
    
    print("\n" + "="*70)
    print("CONTINUING TRAINING")
    print("="*70)
    
    # Training loop
    for epoch in range(start_epoch, config["num_epochs"]):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, metrics_calculator, device
        )
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"  UAR:        {val_metrics['uar']:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_uar"].append(val_metrics["uar"])
        history["epoch_time"].append(epoch_time)
        
        # Save checkpoint
        is_best = val_metrics["uar"] > best_uar
        if is_best:
            best_uar = val_metrics["uar"]
            print(f"  [BEST] New best UAR: {best_uar:.4f}")
        
        if (epoch + 1) % config["save_every"] == 0 or is_best:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "best_uar": best_uar,
                "config": config,
            }
            
            if is_best:
                torch.save(ckpt, save_dir / "best_model.pt")
            
            torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"  [SAVED] Checkpoint saved")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # ETA
        if len(history["epoch_time"]) > 0:
            avg_time = np.mean(history["epoch_time"][-5:])
            remaining = config["num_epochs"] - (epoch + 1)
            eta_min = (avg_time * remaining) / 60
            print(f"  ETA: {eta_min:.1f} minutes")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best UAR: {best_uar:.4f}")


if __name__ == "__main__":
    main()
