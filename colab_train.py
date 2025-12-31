"""
Multimodal FER Training on Google Colab
Convert to notebook: jupytext --to notebook colab_train.py

Or copy cells directly to Colab
"""

# %% [markdown]
# # 🎭 Multimodal FER Training on Google Colab
# 
# Train emotion recognition model on RAVDESS dataset with GPU acceleration.
# 
# **Hardware**: T4 GPU (Free) or A100 (Colab Pro)
# **Dataset**: RAVDESS (1440 videos)
# **Expected Time**: 2-4 hours

# %% [markdown]
# ## 📋 Step 1: Check GPU

# %%
# Check GPU availability
!nvidia-smi

import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [markdown]
# ## 📦 Step 2: Clone Repository from GitHub

# %%
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
%cd multimodal-fer

# Or if already cloned, just pull latest
# !git pull origin main

# %% [markdown]
# ## 💾 Step 3: Mount Google Drive (for RAVDESS data)

# %%
from google.colab import drive
drive.mount('/content/drive')

# Link to your RAVDESS data on Drive
import os
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"  # Adjust this path

# Create symlink to data
!ln -s {RAVDESS_PATH} data/ravdess
!ls -la data/ravdess | head -20

# %% [markdown]
# ## 🔧 Step 4: Install Dependencies

# %%
# Install required packages
!pip install -q transformers==4.36.0
!pip install -q einops
!pip install -q scikit-learn
!pip install -q matplotlib seaborn

# Verify ffmpeg (should be pre-installed on Colab)
!which ffmpeg

print("\n✓ All dependencies installed!")

# %% [markdown]
# ## 📚 Step 5: Import Libraries

# %%
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm.notebook import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Import project modules
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.ravdess_dataset import create_ravdess_dataloaders

print("✓ All imports successful!")

# %% [markdown]
# ## ⚙️ Step 6: Configuration
# 
# **Edit this section** to customize training

# %%
CONFIG = {
    # ============ DATA ============
    "data_dir": "data/ravdess",
    "modality": "speech",  # "speech" or "song"
    "use_audio": True,     # Extract audio from videos
    
    # ============ TRAINING ============
    "batch_size": 16,      # T4: 8-16, A100: 32-64
    "num_epochs": 100,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 2,
    
    # ============ MODEL (Lightweight for T4) ============
    "audio_dim": 512,
    "visual_dim": 512,
    "fusion_hidden_dim": 1024,
    "fusion_output_dim": 512,
    "num_audio_layers": 8,    # Reduced from 17
    "num_visual_layers": 4,   # Reduced from 6
    "num_fusion_layers": 4,   # Reduced from 6
    
    # ============ PRETRAINED MODELS ============
    "use_pretrained_visual": False,  # SigLIP2 (slower, better)
    "use_pretrained_fusion": False,  # LFM2-700M (slower, better)
    
    # ============ OPTIMIZATION ============
    "use_amp": True,                    # Mixed precision (FP16)
    "gradient_accumulation_steps": 1,  # Increase if OOM
    "max_grad_norm": 1.0,
    
    # ============ SCHEDULER ============
    "scheduler_type": "cosine",
    "warmup_epochs": 5,
    
    # ============ CHECKPOINTING ============
    "save_dir": "checkpoints/ravdess_speech_t4",
    "save_every": 10,
    "save_best_only": True,
    
    # ============ DEVICE ============
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Print configuration
print("="*70)
print("TRAINING CONFIGURATION")
print("="*70)
for key, value in CONFIG.items():
    print(f"  {key:<30}: {value}")
print("="*70)

# %% [markdown]
# ## 🏗️ Step 7: Create Model

# %%
def create_model(config):
    """Create model based on configuration."""
    
    audio_config = AudioBranchConfig(
        feature_dim=config["audio_dim"],
        num_layers=config["num_audio_layers"],
        num_segments=8,
    )
    
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=config["use_pretrained_visual"],
        feature_dim=config["visual_dim"],
        temporal_depth=config["num_visual_layers"],
    )
    
    fusion_config = LFM2FusionConfig(
        use_pretrained=config["use_pretrained_fusion"],
        num_layers=config["num_fusion_layers"],
        hidden_dim=config["fusion_hidden_dim"],
        audio_dim=config["audio_dim"],
        visual_dim=config["visual_dim"],
        output_dim=config["fusion_output_dim"],
    )
    
    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        num_classes=8,
        num_segments=8,
    )
    
    return model

print("Creating model...")
model = create_model(CONFIG)
model = model.to(CONFIG["device"])

# Print model summary
model.print_summary()

# %% [markdown]
# ## 📊 Step 8: Create Dataloaders

# %%
print("Creating dataloaders...")

train_loader, val_loader, test_loader = create_ravdess_dataloaders(
    data_dir=CONFIG["data_dir"],
    modality=CONFIG["modality"],
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
    use_audio=CONFIG["use_audio"],
)

print(f"\n✓ Dataloaders created:")
print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
print(f"  Test:  {len(test_loader.dataset)} samples ({len(test_loader)} batches)")

# %% [markdown]
# ## 🎯 Step 9: Training Setup

# %%
# Loss function
criterion = EmotionLoss(
    num_classes=8,
    label_smoothing=0.1,
).to(CONFIG["device"])

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

# Scheduler
if CONFIG["scheduler_type"] == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

# Mixed precision scaler
scaler = GradScaler() if CONFIG["use_amp"] else None

# Metrics calculator
metrics_calculator = EmotionMetrics(num_classes=8)

# Training history
history = {
    "train_loss": [], "val_loss": [],
    "val_accuracy": [], "val_uar": [],
    "val_war": [], "val_wa_f1": [],
    "learning_rate": [],
}

best_uar = 0.0

print("✓ Training setup complete!")

# %% [markdown]
# ## 🔄 Step 10: Training Functions

# %%
def train_epoch(model, train_loader, criterion, optimizer, scaler, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (audio, video, labels, _) in enumerate(pbar):
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])
        
        # Forward
        if config["use_amp"]:
            with autocast(device_type='cuda'):
                outputs = model(audio, video)
                loss = criterion(outputs["logits"], labels)
        else:
            outputs = model(audio, video)
            loss = criterion(outputs["logits"], labels)
        
        # Backward
        if config["use_amp"]:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, metrics_calculator, config):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    for audio, video, labels, _ in pbar:
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])
        
        outputs = model(audio, video)
        loss = criterion(outputs["logits"], labels)
        predictions = outputs["probabilities"].argmax(dim=1)
        
        total_loss += loss.item()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = metrics_calculator.compute(all_predictions, all_labels)
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics

print("✓ Training functions defined!")

# %% [markdown]
# ## 🚀 Step 11: Main Training Loop
# 
# **This will take 2-4 hours on T4 GPU**

# %%
# Create save directory
save_dir = Path(CONFIG["save_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Device: {CONFIG['device']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Save dir: {save_dir}")
print("="*70)

# Training loop
for epoch in range(CONFIG["num_epochs"]):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, CONFIG)
    
    # Validate
    val_metrics = validate(model, val_loader, criterion, metrics_calculator, CONFIG)
    
    # Update scheduler
    scheduler.step()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Save history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_metrics["loss"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["val_uar"].append(val_metrics["uar"])
    history["val_war"].append(val_metrics["war"])
    history["val_wa_f1"].append(val_metrics["wa_f1"])
    history["learning_rate"].append(optimizer.param_groups[0]["lr"])
    
    # Print metrics
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_metrics['loss']:.4f}")
    print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"  UAR:        {val_metrics['uar']:.4f} ⭐")
    print(f"  WAR:        {val_metrics['war']:.4f}")
    print(f"  WA-F1:      {val_metrics['wa_f1']:.4f}")
    print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    is_best = val_metrics["uar"] > best_uar
    if is_best:
        best_uar = val_metrics["uar"]
        print(f"\n  🎉 New best UAR: {best_uar:.4f}")
        
        # Save best checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": val_metrics,
            "history": history,
            "config": CONFIG,
        }
        torch.save(checkpoint, save_dir / "best_model.pth")
        print(f"  ✓ Best model saved")
    
    # Save periodic checkpoint
    if (epoch + 1) % CONFIG["save_every"] == 0:
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")

# Save final model
final_checkpoint = {
    "epoch": CONFIG["num_epochs"] - 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "metrics": val_metrics,
    "history": history,
    "config": CONFIG,
}
torch.save(final_checkpoint, save_dir / "final_model.pth")

# Save training history
with open(save_dir / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Best UAR: {best_uar:.4f}")
print(f"Checkpoints saved to: {save_dir}")

# %% [markdown]
# ## 📈 Step 12: Plot Training Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
axes[0, 0].plot(history["val_loss"], label="Val", linewidth=2)
axes[0, 0].set_xlabel("Epoch", fontsize=12)
axes[0, 0].set_ylabel("Loss", fontsize=12)
axes[0, 0].set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# UAR
axes[0, 1].plot(history["val_uar"], label="UAR", color="green", linewidth=2)
axes[0, 1].axhline(y=best_uar, color='r', linestyle='--', linewidth=2, label=f'Best: {best_uar:.4f}')
axes[0, 1].set_xlabel("Epoch", fontsize=12)
axes[0, 1].set_ylabel("UAR", fontsize=12)
axes[0, 1].set_title("Validation UAR (Primary Metric)", fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# All metrics
axes[1, 0].plot(history["val_accuracy"], label="Accuracy", linewidth=2)
axes[1, 0].plot(history["val_uar"], label="UAR", linewidth=2)
axes[1, 0].plot(history["val_war"], label="WAR", linewidth=2)
axes[1, 0].plot(history["val_wa_f1"], label="WA-F1", linewidth=2)
axes[1, 0].set_xlabel("Epoch", fontsize=12)
axes[1, 0].set_ylabel("Score", fontsize=12)
axes[1, 0].set_title("All Validation Metrics", fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Learning rate
axes[1, 1].plot(history["learning_rate"], color="orange", linewidth=2)
axes[1, 1].set_xlabel("Epoch", fontsize=12)
axes[1, 1].set_ylabel("Learning Rate", fontsize=12)
axes[1, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Training curves saved to: {save_dir}/training_curves.png")

# %% [markdown]
# ## 🧪 Step 13: Evaluate on Test Set

# %%
print("="*70)
print("EVALUATING ON TEST SET")
print("="*70)

# Load best model
best_checkpoint = torch.load(save_dir / "best_model.pth")
model.load_state_dict(best_checkpoint["model_state_dict"])

# Evaluate
test_metrics = validate(model, test_loader, criterion, metrics_calculator, CONFIG)

print("\n📊 Test Set Results:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  UAR:      {test_metrics['uar']:.4f} ⭐")
print(f"  WAR:      {test_metrics['war']:.4f}")
print(f"  WA-F1:    {test_metrics['wa_f1']:.4f}")

# Print per-class metrics
print("\n📋 Per-Class Metrics:")
metrics_calculator.print_metrics(test_metrics)

# Save test results
test_results = {
    "test_metrics": test_metrics,
    "best_epoch": best_checkpoint["epoch"] + 1,
    "best_val_uar": best_checkpoint["metrics"]["uar"],
}

with open(save_dir / "test_results.json", 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✓ Test results saved to: {save_dir}/test_results.json")

# %% [markdown]
# ## 💾 Step 14: Download Checkpoints

# %%
# Download files to local machine
from google.colab import files

print("Downloading checkpoints...")

# Download best model
files.download(str(save_dir / "best_model.pth"))

# Download training history
files.download(str(save_dir / "training_history.json"))

# Download test results
files.download(str(save_dir / "test_results.json"))

# Download training curves
files.download(str(save_dir / "training_curves.png"))

print("✓ All files downloaded!")

# %% [markdown]
# ## 🎉 Training Complete!
# 
# ### Next Steps:
# 
# 1. **Download checkpoints** (already done above)
# 2. **Test on local machine**:
#    ```bash
#    python scripts/inference_cpu.py
#    ```
# 3. **Push to GitHub**:
#    ```bash
#    git add checkpoints/
#    git commit -m "Add trained model"
#    git push
#    ```
# 
# ### Expected Results:
# - **UAR**: 75-80% (lightweight) or 80-85% (pretrained)
# - **Training time**: 2-4 hours on T4
# - **Model size**: ~150M parameters
# 
# **Congratulations! 🎊**
