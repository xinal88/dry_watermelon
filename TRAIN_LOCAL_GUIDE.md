# 🎯 Local Training Guide

Hướng dẫn train model trên máy local hoặc qua Kiro IDE kết nối với Colab.

## 📋 Yêu cầu

1. **Dữ liệu**: RAVDESS dataset trong folder `data/ravdess/`
2. **GPU**: CUDA-compatible GPU (khuyến nghị)
3. **Dependencies**: Đã cài đặt tất cả packages

## 🚀 Cách 1: Chạy Training Script

### Quick Start

```bash
# Basic training
python scripts/train_ravdess.py --data_dir data/ravdess --epochs 100

# Custom configuration
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir checkpoints/my_experiment
```

### Tham số quan trọng

```bash
# Data
--data_dir data/ravdess          # Đường dẫn đến dữ liệu
--modality speech                # "speech" hoặc "song"
--use_audio                      # Sử dụng audio modality

# Training
--batch_size 8                   # Batch size (8-16 cho GPU nhỏ)
--epochs 100                     # Số epochs
--lr 1e-4                        # Learning rate
--num_workers 2                  # DataLoader workers

# Model (Lightweight cho GPU nhỏ)
--audio_dim 512
--visual_dim 512
--fusion_hidden_dim 1024
--num_audio_layers 8
--num_visual_layers 4
--num_fusion_layers 4

# Optimization
--use_amp                        # Mixed precision (FP16)
--max_grad_norm 1.0             # Gradient clipping

# Checkpointing
--save_dir checkpoints/ravdess_local
--save_every 10                  # Lưu mỗi 10 epochs
--resume checkpoints/xxx.pt      # Resume từ checkpoint
```

## 🔧 Cách 2: Test trước khi Train

### 1. Kiểm tra dữ liệu

```python
from pathlib import Path

data_path = Path("data/ravdess")
print(f"Data exists: {data_path.exists()}")

speech_folders = list(data_path.glob("Video_Speech_Actor_*"))
print(f"Found {len(speech_folders)} actors")
```

### 2. Test dataloader

```python
from data.ravdess_dataset import create_ravdess_dataloaders

train_loader, val_loader, test_loader = create_ravdess_dataloaders(
    data_dir="data/ravdess",
    modality="speech",
    batch_size=4,
    num_workers=0,  # 0 for debugging
    use_audio=True,
)

print(f"Train: {len(train_loader.dataset)} samples")
print(f"Val: {len(val_loader.dataset)} samples")
print(f"Test: {len(test_loader.dataset)} samples")

# Test one batch
audio, video, labels, metadata = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  Audio: {audio.shape}")
print(f"  Video: {video.shape}")
print(f"  Labels: {labels.shape}")
```

### 3. Test model

```python
import torch
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig

# Create lightweight model
audio_config = AudioBranchConfig(feature_dim=512, num_layers=8)
visual_config = VisualBranchConfig(feature_dim=512, temporal_depth=4)
fusion_config = LFM2FusionConfig(
    num_layers=4,
    hidden_dim=1024,
    audio_dim=512,
    visual_dim=512,
    output_dim=512,
)

model = MultimodalFER(
    audio_config=audio_config,
    visual_config=visual_config,
    fusion_config=fusion_config,
    num_classes=8,
)

model.print_summary()

# Test forward pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

audio = torch.randn(2, 48000).to(device)  # 2 samples, 3 seconds @ 16kHz
video = torch.randn(2, 16, 3, 224, 224).to(device)  # 2 samples, 16 frames

outputs = model(audio, video)
print(f"\nOutput shape: {outputs.shape}")  # Should be [2, 8]
```

## 📊 Monitoring Training

### Xem training progress

```python
import json
from pathlib import Path

# Load history
history_path = Path("checkpoints/ravdess_local/history.json")
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Best UAR: {max(history['val_uar']):.4f}")
    print(f"Latest train loss: {history['train_loss'][-1]:.4f}")
    print(f"Latest val loss: {history['val_loss'][-1]:.4f}")
```

### Visualize training curves

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training Loss')

# UAR
axes[1].plot(history['val_uar'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('UAR')
axes[1].set_title('Validation UAR')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

## 🔄 Resume Training

```bash
# Resume từ checkpoint cuối cùng
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --resume checkpoints/ravdess_local/checkpoint_epoch_50.pt \
    --epochs 100
```

## 💡 Tips

### Nếu gặp Out of Memory (OOM)

1. Giảm batch size: `--batch_size 4`
2. Giảm số workers: `--num_workers 0`
3. Giảm model size:
   ```bash
   --num_audio_layers 6 \
   --num_visual_layers 3 \
   --num_fusion_layers 3 \
   --fusion_hidden_dim 512
   ```

### Tăng tốc training

1. Tăng batch size (nếu có đủ VRAM): `--batch_size 16`
2. Sử dụng mixed precision: `--use_amp`
3. Tăng num_workers: `--num_workers 4`

### Debug mode

```bash
# Train với 1 epoch để test
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --batch_size 2 \
    --epochs 1 \
    --num_workers 0 \
    --save_dir checkpoints/debug
```

## 📁 Output Structure

```
checkpoints/ravdess_local/
├── config.json                    # Training configuration
├── history.json                   # Training history
├── best_model.pt                  # Best model (highest UAR)
├── checkpoint_epoch_10.pt         # Checkpoint at epoch 10
├── checkpoint_epoch_20.pt         # Checkpoint at epoch 20
└── ...
```

## 🎯 Expected Results

Với RAVDESS speech dataset:

- **Train samples**: ~960 videos (actors 1-16)
- **Val samples**: ~240 videos (actors 17-20)
- **Test samples**: ~240 videos (actors 21-24)

Expected performance sau 100 epochs:
- **UAR**: 0.65-0.75
- **Accuracy**: 0.70-0.80
- **Training time**: 2-4 giờ (T4 GPU)

## 🐛 Troubleshooting

### Lỗi: "Loaded 0 videos"

```python
# Check data path
from pathlib import Path
data_path = Path("data/ravdess")
print(f"Exists: {data_path.exists()}")
print(f"Folders: {list(data_path.glob('Video_Speech_Actor_*'))[:3]}")
```

### Lỗi: "CUDA out of memory"

Giảm batch size hoặc model size (xem Tips trên).

### Lỗi: "ffmpeg not found"

```bash
# Install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 📞 Support

Nếu gặp vấn đề, check:
1. `TEST_STATUS.md` - Test results
2. `TRAINING_GUIDE.md` - Detailed training guide
3. `QUICK_REFERENCE.md` - Quick commands
