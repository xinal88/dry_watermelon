"""
Quick Training Script - Half Dataset for RTX 3050

Train on 50% of RAVDESS dataset with optimized settings for 4GB VRAM.
Expected time: 1-2 hours for 50 epochs

Usage:
    python scripts/train_half_dataset.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import time
from datetime import datetime
import numpy as np

# Import project modules
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.ravdess_dataset import create_ravdess_dataloaders


def create_half_dataloaders(data_dir, modality="speech", batch_size=4, num_workers=2):
    """Create dataloaders with 50% of data."""
    print("Creating dataloaders with 50% dataset...")
    
    # Get full dataloaders with num_workers=0 to avoid memory issues
    train_loader, val_loader, test_loader = create_ravdess_dataloaders(
        data_dir=data_dir,
        modality=modality,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing memory issues
        use_audio=True,
    )
    
    # Use 50% of training data (keep all val/test for proper evaluation)
    train_dataset = train_loader.dataset
    train_size = len(train_dataset)
    half_size = train_size // 2
    
    # Random subset
    indices = np.random.permutation(train_size)[:half_size]
    half_train_dataset = Subset(train_dataset, indices)
    
    # Create new train loader with half data
    half_train_loader = DataLoader(
        half_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers= 0,  # Set to 0
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"[OK] Using 50% of training data:")
    print(f"  Train: {len(half_train_loader.dataset)} samples (was {train_size})")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    return half_train_loader, val_loader, test_loader


def create_lightweight_model():
    """Create lightweight model for RTX 3050."""
    print("\nCreating lightweight model for RTX 3050...")
    
    # Even smaller model for 4GB VRAM
    audio_config = AudioBranchConfig(
        feature_dim=128,      # Reduced from 256
        num_layers=4,         # Reduced from 6
        num_segments=8,
    )
    
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=False,
        feature_dim=128,      # Reduced from 256
        temporal_depth=2,     # Reduced from 3
    )
    
    fusion_config = LFM2FusionConfig(
        use_pretrained=False,
        num_layers=2,         # Reduced from 3
        hidden_dim=256,       # Reduced from 512
        audio_dim=128,
        visual_dim=128,
        output_dim=128,       # Reduced from 256
    )
    
    # Import ClassifierConfig
    from models.classifier import ClassifierConfig
    
    # Classifier config matching fusion output
    classifier_config = ClassifierConfig(
        input_dim=128,        # Match fusion output_dim
        hidden_dims=[128, 64],  # Smaller hidden layers
        num_classes=8,
        dropout=0.1,
    )
    
    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        classifier_config=classifier_config,  # Add classifier config
        num_classes=8,
        num_segments=8,
    )
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for audio, video, labels, _ in pbar:
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(audio, video)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss detected, skipping batch")
            continue
        
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        
        # Clip gradients more aggressively
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Check gradients
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if grad_norm > 100:  # Skip if gradients explode
            print(f"\nWarning: Large gradient norm {grad_norm:.2f}, skipping")
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"})
    
    if valid_batches == 0:
        return float('inf')
    
    return total_loss / valid_batches


def validate(model, val_loader, criterion, metrics_calculator, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, video, labels, _ in tqdm(val_loader, desc="Validation", ncols=100):
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            
            outputs = model(audio, video)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = metrics_calculator.compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, metrics


def main():
    """Main training function."""
    print("="*70)
    print("QUICK TRAINING - 50% DATASET - RTX 3050")
    print("="*70)
    
    # Configuration optimized for RTX 3050
    CONFIG = {
        "data_dir": "data/ravdess",
        "modality": "speech",
        "batch_size": 2,          # Very small batch for 4GB VRAM
        "num_epochs": 7,         # Fewer epochs for quick training
        "lr": 5e-5,               # Lower LR to prevent divergence
        "weight_decay": 0.01,
        "num_workers": 0,         # Set to 0 to avoid memory issues
        "save_dir": "checkpoints/half_dataset_rtx3050",
        "save_every": 10,
    }
    
    # Print config
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key:<20}: {value}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create save directory
    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create dataloaders (50% training data)
    train_loader, val_loader, test_loader = create_half_dataloaders(
        data_dir=CONFIG["data_dir"],
        modality=CONFIG["modality"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    
    # Create lightweight model
    model = create_lightweight_model()
    model = model.to(device)
    model.print_summary()
    
    # Optimizer
    criterion = EmotionLoss(num_classes=8, label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        eps=1e-8,  # Add epsilon for numerical stability
    )
    
    # Scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Metrics
    metrics_calculator = EmotionMetrics(num_classes=8)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_uar": [],
        "epoch_time": [],
    }
    
    best_uar = 0.0
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, metrics_calculator, device
        )
        
        # Update schedulers
        if epoch < 5:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log results
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
        
        if (epoch + 1) % CONFIG["save_every"] == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "best_uar": best_uar,
                "config": CONFIG,
            }
            
            if is_best:
                torch.save(checkpoint, save_dir / "best_model.pt")
            
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"  [SAVED] Checkpoint saved")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Estimate remaining time
        avg_epoch_time = np.mean(history["epoch_time"])
        remaining_epochs = CONFIG["num_epochs"] - (epoch + 1)
        eta_minutes = (avg_epoch_time * remaining_epochs) / 60
        print(f"  ETA: {eta_minutes:.1f} minutes")
    
    total_time = time.time() - start_time
    
    # Final test
    print("\n" + "="*70)
    print("TESTING ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_metrics = validate(
        model, test_loader, criterion, metrics_calculator, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  UAR:      {test_metrics['uar']:.4f}")
    print(f"  WAR:      {test_metrics['war']:.4f}")
    print(f"  WA-F1:    {test_metrics['wa_f1']:.4f}")
    
    # Save test results
    test_results = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_val_uar": best_uar,
        "total_training_time_minutes": total_time / 60,
    }
    
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val UAR:  {best_uar:.4f}")
    print(f"Test UAR:      {test_metrics['uar']:.4f}")
    print(f"Total time:    {total_time/60:.1f} minutes")
    print(f"Checkpoints:   {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
