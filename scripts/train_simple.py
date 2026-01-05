"""
Simple Training Script - No Mixed Precision

Simplified version without AMP to avoid NaN issues.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import time
import numpy as np

from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from models.classifier import ClassifierConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.ravdess_dataset import create_ravdess_dataloaders


def create_model():
    """Create ultra-lightweight model."""
    print("\nCreating model...")
    
    audio_config = AudioBranchConfig(feature_dim=128, num_layers=4, num_segments=8)
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=False, feature_dim=128, temporal_depth=2
    )
    fusion_config = LFM2FusionConfig(
        use_pretrained=False, num_layers=2, hidden_dim=256,
        audio_dim=128, visual_dim=128, output_dim=128,
    )
    classifier_config = ClassifierConfig(
        input_dim=128, hidden_dims=[128, 64], num_classes=8, dropout=0.1,
    )
    
    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        classifier_config=classifier_config,
        num_classes=8,
        num_segments=8,
    )
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch - NO MIXED PRECISION."""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for audio, video, labels, _ in pbar:
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - NO AMP
        outputs = model(audio, video)
        if isinstance(outputs, dict):
            outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
        
        loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nSkipping NaN batch")
            continue
        
        # Backward
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(valid_batches, 1)


def validate(model, val_loader, criterion, metrics_calc, device):
    """Validate."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
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
    
    metrics = metrics_calc.compute_metrics(all_labels, all_preds)
    return total_loss / len(val_loader), metrics


def main():
    print("="*70)
    print("SIMPLE TRAINING - NO MIXED PRECISION")
    print("="*70)
    
    CONFIG = {
        "data_dir": "data/ravdess",
        "batch_size": 2,
        "num_epochs": 30,
        "lr": 1e-5,  # Very low LR
        "weight_decay": 0.01,
    }
    
    print("\nConfig:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_ravdess_dataloaders(
        data_dir=CONFIG["data_dir"],
        modality="speech",
        batch_size=CONFIG["batch_size"],
        num_workers=0,
        use_audio=True,
    )
    
    # Use 50% training data
    train_dataset = train_loader.dataset
    half_size = len(train_dataset) // 2
    indices = np.random.permutation(len(train_dataset))[:half_size]
    half_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(
        half_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    
    # Model
    model = create_model()
    model = model.to(device)
    model.print_summary()
    
    # Training setup
    criterion = EmotionLoss(num_classes=8, label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"], eps=1e-8
    )
    
    metrics_calc = EmotionMetrics(num_classes=8)
    
    save_dir = Path("checkpoints/simple_training")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "val_loss": [], "val_uar": []}
    best_uar = 0.0
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-"*70)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics_calc, device)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"  UAR:        {val_metrics['uar']:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_uar"].append(val_metrics["uar"])
        
        is_best = val_metrics["uar"] > best_uar
        if is_best:
            best_uar = val_metrics["uar"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_uar": best_uar,
            }, save_dir / "best_model.pt")
            print(f"  [BEST] New best UAR: {best_uar:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_uar": best_uar,
            }, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best UAR: {best_uar:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
