"""
Training script for RAVDESS dataset - Local/IDE version

Usage:
    python scripts/train_ravdess.py --data_dir data/ravdess --epochs 100
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import argparse
from datetime import datetime

# Import project modules
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.ravdess_dataset import create_ravdess_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal FER on RAVDESS")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/ravdess", help="Path to RAVDESS data")
    parser.add_argument("--modality", type=str, default="speech", choices=["speech", "song"])
    parser.add_argument("--use_audio", action="store_true", default=True, help="Use audio modality")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    
    # Model
    parser.add_argument("--audio_dim", type=int, default=512)
    parser.add_argument("--visual_dim", type=int, default=512)
    parser.add_argument("--fusion_hidden_dim", type=int, default=1024)
    parser.add_argument("--fusion_output_dim", type=int, default=512)
    parser.add_argument("--num_audio_layers", type=int, default=8)
    parser.add_argument("--num_visual_layers", type=int, default=4)
    parser.add_argument("--num_fusion_layers", type=int, default=4)
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/ravdess_local")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    audio_config = AudioBranchConfig(
        feature_dim=args.audio_dim,
        num_layers=args.num_audio_layers,
        num_segments=8,
    )
    
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=False,
        feature_dim=args.visual_dim,
        temporal_depth=args.num_visual_layers,
    )
    
    fusion_config = LFM2FusionConfig(
        use_pretrained=False,
        num_layers=args.num_fusion_layers,
        hidden_dim=args.fusion_hidden_dim,
        audio_dim=args.audio_dim,
        visual_dim=args.visual_dim,
        output_dim=args.fusion_output_dim,
    )
    
    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        num_classes=8,
        num_segments=8,
    )
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (audio, video, labels, _) in enumerate(pbar):
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(audio, video)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(audio, video)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, metrics_calculator, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, video, labels, _ in tqdm(val_loader, desc="Validation"):
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            
            outputs = model(audio, video)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = metrics_calculator.compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, metrics


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_ravdess_dataloaders(
        data_dir=args.data_dir,
        modality=args.modality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_audio=args.use_audio,
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")
    print(f"Test:  {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args)
    model = model.to(device)
    model.print_summary()
    
    # Loss and optimizer
    criterion = EmotionLoss(num_classes=8, label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler() if args.use_amp else None
    
    # Metrics
    metrics_calculator = EmotionMetrics(num_classes=8)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_uar": [],
    }
    
    best_uar = 0.0
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_uar = checkpoint.get("best_uar", 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, metrics_calculator, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"  UAR:        {val_metrics['uar']:.4f}")
        print(f"  WAR:        {val_metrics['war']:.4f}")
        print(f"  WA-F1:      {val_metrics['wa_f1']:.4f}")
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_uar"].append(val_metrics["uar"])
        
        # Save checkpoint
        is_best = val_metrics["uar"] > best_uar
        if is_best:
            best_uar = val_metrics["uar"]
            print(f"  ✓ New best UAR: {best_uar:.4f}")
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "best_uar": best_uar,
                "config": vars(args),
            }
            
            if is_best:
                torch.save(checkpoint, save_dir / "best_model.pt")
            
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"  ✓ Checkpoint saved")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best UAR: {best_uar:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
