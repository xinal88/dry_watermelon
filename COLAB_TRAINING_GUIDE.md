# 🚀 Hướng Dẫn Train trên Google Colab Pro

## 📋 Tổng Quan

Hướng dẫn chi tiết để train Multimodal FER model trên Google Colab Pro với GPU mạnh (T4/A100).

---

## 🎯 Bước 1: Chuẩn Bị Dữ Liệu trên Google Drive

### 1.1. Cấu Trúc Thư Mục trên Drive

```
My Drive/
└── RAVDESS_Multimodal_FER/
    ├── data/
    │   └── ravdess/
    │       ├── Video_Speech_Actor_01/
    │       ├── Video_Speech_Actor_02/
    │       ├── ...
    │       ├── Video_Speech_Actor_24/
    │       ├── Video_Song_Actor_01/
    │       └── ...
    └── checkpoints/  (sẽ tự tạo)
```

### 1.2. Upload Code lên Drive

**Option A: Upload thủ công**
- Nén toàn bộ project: `dry_watermelon.zip`
- Upload lên Drive: `My Drive/RAVDESS_Multimodal_FER/`
- Giải nén trên Colab

**Option B: Sync từ GitHub** (khuyến nghị)
- Push code lên GitHub repository
- Clone trực tiếp trên Colab

---

## 🎯 Bước 2: Tạo Colab Notebook

Tạo file mới: `Train_Multimodal_FER.ipynb` trên Colab

---

## 📝 Nội Dung Notebook

### Cell 1: Kiểm Tra GPU

```python
# Kiểm tra GPU
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn đến project
PROJECT_DIR = "/content/drive/MyDrive/RAVDESS_Multimodal_FER"
%cd {PROJECT_DIR}
```

### Cell 3: Cài Đặt Dependencies

```python
# Cài đặt các thư viện cần thiết
!pip install -q transformers==4.36.0
!pip install -q einops
!pip install -q tqdm
!pip install -q scikit-learn
!pip install -q matplotlib
!pip install -q seaborn

# Kiểm tra ffmpeg (Colab đã có sẵn)
!ffmpeg -version | head -n 1
```

### Cell 4: Import Libraries

```python
import sys
sys.path.insert(0, PROJECT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm.notebook import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import model components
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.ravdess_dataset import create_ravdess_dataloaders

print("✓ All imports successful!")
```

### Cell 5: Configuration

```python
# ============================================================================
# 🎯 CONFIGURATION - EDIT THIS SECTION
# ============================================================================

CONFIG = {
    # Data
    "data_dir": f"{PROJECT_DIR}/data/ravdess",
    "modality": "speech",  # "speech" or "song"
    "use_audio": True,  # Set False if ffmpeg issues
    
    # Training
    "batch_size": 16,  # Colab Pro có thể dùng 16-32
    "num_epochs": 100,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 2,
    
    # Model - LIGHTWEIGHT for faster training
    "audio_dim": 512,
    "visual_dim": 512,
    "fusion_hidden_dim": 1024,
    "fusion_output_dim": 512,
    "num_audio_layers": 8,  # Reduced from 17
    "num_visual_layers": 4,  # Reduced from 6
    "num_fusion_layers": 4,  # Reduced from 6
    
    # Pretrained models
    "use_pretrained_visual": False,  # Set True for SigLIP2 (slower)
    "use_pretrained_fusion": False,  # Set True for LFM2-700M (slower)
    
    # Optimization
    "use_amp": True,  # Mixed precision
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    
    # Scheduler
    "scheduler_type": "cosine",  # "cosine" or "step"
    "warmup_epochs": 5,
    
    # Checkpointing
    "save_dir": f"{PROJECT_DIR}/checkpoints/ravdess_speech",
    "save_every": 10,  # Save every N epochs
    "save_best_only": True,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
```

### Cell 6: Create Model

```python
def create_model(config):
    """Create model based on configuration."""
    
    # Model configs
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

# Create model
print("Creating model...")
model = create_model(CONFIG)
model = model.to(CONFIG["device"])

# Print summary
model.print_summary()
```

### Cell 7: Create Dataloaders

```python
print("Creating dataloaders...")

train_loader, val_loader, test_loader = create_ravdess_dataloaders(
    data_dir=CONFIG["data_dir"],
    modality=CONFIG["modality"],
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
    use_audio=CONFIG["use_audio"],
)

print(f"✓ Train: {len(train_loader.dataset)} samples")
print(f"✓ Val: {len(val_loader.dataset)} samples")
print(f"✓ Test: {len(test_loader.dataset)} samples")
```

### Cell 8: Training Setup

```python
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
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1,
    )

# Mixed precision scaler
scaler = GradScaler() if CONFIG["use_amp"] else None

# Metrics calculator
metrics_calculator = EmotionMetrics(num_classes=8)

# Training history
history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_uar": [],
    "val_war": [],
    "val_wa_f1": [],
    "learning_rate": [],
}

# Best model tracking
best_uar = 0.0

print("✓ Training setup complete!")
```

### Cell 9: Training Functions

```python
def train_epoch(model, train_loader, criterion, optimizer, scaler, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (audio, video, labels, _) in enumerate(pbar):
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])
        
        # Forward with mixed precision
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
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, metrics_calculator, config):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for audio, video, labels, _ in pbar:
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])
        
        # Forward
        outputs = model(audio, video)
        loss = criterion(outputs["logits"], labels)
        
        # Get predictions
        predictions = outputs["probabilities"].argmax(dim=1)
        
        # Accumulate
        total_loss += loss.item()
        num_batches += 1
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_calculator.compute(
        predictions=all_predictions,
        labels=all_labels,
    )
    
    metrics["loss"] = total_loss / num_batches
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, history, save_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "history": history,
        "config": CONFIG,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)

print("✓ Training functions defined!")
```

### Cell 10: Main Training Loop

```python
# Create save directory
save_dir = Path(CONFIG["save_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TRAINING MULTIMODAL FER ON RAVDESS")
print("="*70)
print(f"Device: {CONFIG['device']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print("="*70)

# Training loop
for epoch in range(CONFIG["num_epochs"]):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    
    # Train
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, scaler, CONFIG
    )
    
    # Validate
    val_metrics = validate(
        model, val_loader, criterion, metrics_calculator, CONFIG
    )
    
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
        print(f"  ✓ New best UAR: {best_uar:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % CONFIG["save_every"] == 0 or is_best:
        save_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, history, save_path, is_best
        )
        print(f"  ✓ Checkpoint saved: {save_path.name}")

# Save final model
final_path = save_dir / "final_model.pth"
save_checkpoint(
    model, optimizer, scheduler, CONFIG["num_epochs"]-1, val_metrics, history, final_path
)

# Save training history
history_path = save_dir / "training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Best UAR: {best_uar:.4f}")
print(f"Checkpoints saved to: {save_dir}")
```

### Cell 11: Plot Training History

```python
# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history["train_loss"], label="Train Loss")
axes[0, 0].plot(history["val_loss"], label="Val Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training and Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# UAR
axes[0, 1].plot(history["val_uar"], label="UAR", color="green")
axes[0, 1].axhline(y=best_uar, color='r', linestyle='--', label=f'Best: {best_uar:.4f}')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("UAR")
axes[0, 1].set_title("Validation UAR (Primary Metric)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# All metrics
axes[1, 0].plot(history["val_accuracy"], label="Accuracy")
axes[1, 0].plot(history["val_uar"], label="UAR")
axes[1, 0].plot(history["val_war"], label="WAR")
axes[1, 0].plot(history["val_wa_f1"], label="WA-F1")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Score")
axes[1, 0].set_title("All Validation Metrics")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Learning rate
axes[1, 1].plot(history["learning_rate"], color="orange")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Learning Rate")
axes[1, 1].set_title("Learning Rate Schedule")
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Training curves saved to: {save_dir}/training_curves.png")
```

### Cell 12: Evaluate on Test Set

```python
print("="*70)
print("EVALUATING ON TEST SET")
print("="*70)

# Load best model
best_checkpoint = torch.load(save_dir / "best_model.pth")
model.load_state_dict(best_checkpoint["model_state_dict"])

# Evaluate
test_metrics = validate(
    model, test_loader, criterion, metrics_calculator, CONFIG
)

print("\nTest Set Results:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  UAR:      {test_metrics['uar']:.4f} ⭐")
print(f"  WAR:      {test_metrics['war']:.4f}")
print(f"  WA-F1:    {test_metrics['wa_f1']:.4f}")

# Print per-class metrics
print("\nPer-Class Metrics:")
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
```

### Cell 13: Download Checkpoints (Optional)

```python
# Download best model to local machine
from google.colab import files

# Download best model
files.download(str(save_dir / "best_model.pth"))

# Download training history
files.download(str(save_dir / "training_history.json"))

# Download test results
files.download(str(save_dir / "test_results.json"))

print("✓ Files downloaded!")
```

---

## 📊 Kết Quả Mong Đợi

### Với Lightweight Config:
- **Training time**: ~2-3 giờ (100 epochs trên T4)
- **Expected UAR**: 75-80%
- **Model size**: ~150M parameters

### Với Full Pretrained Config:
- **Training time**: ~4-6 giờ (100 epochs trên T4)
- **Expected UAR**: 80-85%
- **Model size**: ~400M parameters

---

## 💡 Tips & Tricks

### 1. Tăng Tốc Training:
```python
# Tăng batch size
"batch_size": 32,  # Nếu GPU đủ mạnh

# Giảm số layers
"num_audio_layers": 4,
"num_visual_layers": 2,
"num_fusion_layers": 2,
```

### 2. Cải Thiện Accuracy:
```python
# Dùng pretrained models
"use_pretrained_visual": True,
"use_pretrained_fusion": True,

# Tăng số epochs
"num_epochs": 150,

# Data augmentation (thêm vào dataset)
```

### 3. Xử Lý OOM:
```python
# Giảm batch size
"batch_size": 8,

# Gradient accumulation
"gradient_accumulation_steps": 4,

# Không dùng pretrained
"use_pretrained_visual": False,
"use_pretrained_fusion": False,
```

---

## 🔄 Sync Checkpoints về Local

### Option 1: Download trực tiếp từ Colab
```python
from google.colab import files
files.download("checkpoints/ravdess_speech/best_model.pth")
```

### Option 2: Sync qua Drive
- Checkpoints tự động lưu trên Drive
- Download từ Drive về máy local

### Option 3: Sử dụng rclone
```bash
# Cài rclone trên local
# Sync từ Drive về
rclone copy "drive:RAVDESS_Multimodal_FER/checkpoints" "./checkpoints"
```

---

## ✅ Checklist

- [ ] Upload dữ liệu RAVDESS lên Drive
- [ ] Upload code lên Drive hoặc GitHub
- [ ] Tạo Colab notebook
- [ ] Chọn GPU runtime (T4/A100)
- [ ] Mount Drive
- [ ] Cài đặt dependencies
- [ ] Chạy training
- [ ] Monitor training curves
- [ ] Evaluate trên test set
- [ ] Download checkpoints

---

## 🎉 Hoàn Thành!

Sau khi train xong, bạn có thể:
1. Download `best_model.pth` về máy local
2. Sử dụng `scripts/inference_cpu.py` để test
3. Deploy model cho production

**Good luck with training!** 🚀
