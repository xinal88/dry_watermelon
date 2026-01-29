# 🚀 Quick Start: Train trên Google Colab Pro

## 📋 Tóm Tắt

Hướng dẫn nhanh để train Multimodal FER model trên Google Colab Pro với RAVDESS dataset.

**Thời gian:** ~2-3 giờ (bao gồm setup + training)
**Yêu cầu:** Google Colab Pro (hoặc Pro+)
**Dataset:** RAVDESS (1,440 videos, ~3GB)

---

## 🎯 Bước 1: Setup Colab Notebook

### 1.1. Tạo Notebook Mới

1. Mở [Google Colab](https://colab.research.google.com/)
2. Tạo notebook mới: `File > New notebook`
3. Đổi tên: `Multimodal_FER_Training.ipynb`
4. Chọn runtime: `Runtime > Change runtime type > GPU (A100)`

### 1.2. Mount Google Drive

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
import os
print("✅ Google Drive mounted!")
print(f"Drive contents: {os.listdir('/content/drive/MyDrive')[:5]}")
```

---

## 📦 Bước 2: Upload RAVDESS Dataset

### 2.1. Download RAVDESS

1. Truy cập [RAVDESS Dataset](https://zenodo.org/record/1188976)
2. Download: `Video_Speech_Actor_01.zip` đến `Video_Speech_Actor_24.zip`
3. Hoặc download full: `RAVDESS_full.zip`

### 2.2. Upload lên Google Drive

```
Cấu trúc thư mục:
/content/drive/MyDrive/RAVDESS/
├── Actor_01/
│   ├── 01-01-01-01-01-01-01.mp4
│   ├── 01-01-01-01-01-02-01.mp4
│   └── ...
├── Actor_02/
│   └── ...
└── Actor_24/
    └── ...

Hoặc:
/content/drive/MyDrive/RAVDESS/
├── Video_Speech_Actor_01/
│   ├── 01-01-01-01-01-01-01.mp4
│   └── ...
└── ...
```

### 2.3. Verify Dataset

```python
# Cell 2: Verify RAVDESS dataset
import os
from pathlib import Path

RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"

# Check if path exists
if not os.path.exists(RAVDESS_PATH):
    print(f"❌ ERROR: RAVDESS path not found!")
    print(f"   Expected: {RAVDESS_PATH}")
    print(f"\n   Please upload RAVDESS dataset to Google Drive")
else:
    print(f"✅ RAVDESS path found: {RAVDESS_PATH}")
    
    # Count videos
    video_count = 0
    actor_folders = []
    
    for item in os.listdir(RAVDESS_PATH):
        item_path = os.path.join(RAVDESS_PATH, item)
        if os.path.isdir(item_path) and ("Actor" in item or "Video_Speech_Actor" in item):
            actor_folders.append(item)
            videos = [f for f in os.listdir(item_path) if f.endswith('.mp4')]
            video_count += len(videos)
    
    print(f"✅ Found {len(actor_folders)} actor folders")
    print(f"✅ Found {video_count} videos")
    
    if video_count < 1000:
        print(f"\n⚠️  WARNING: Expected ~1440 videos, found {video_count}")
        print(f"   Please check if all Actor folders are uploaded")
```

---

## 🔧 Bước 3: Clone Repository và Install Dependencies

```python
# Cell 3: Clone repository
!git clone https://github.com/your-username/multimodal-fer.git
%cd multimodal-fer

print("✅ Repository cloned!")
```

```python
# Cell 4: Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers timm einops
!pip install -q opencv-python librosa soundfile
!pip install -q tensorboard wandb
!pip install -q tqdm scikit-learn

print("✅ Dependencies installed!")
```

```python
# Cell 5: Verify GPU
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("❌ WARNING: GPU not available!")
```

---

## 🧪 Bước 4: Test Model và Dataset

```python
# Cell 6: Test complete model
!python tests/test_complete_model.py

# Expected output:
# ✅ Model created successfully
# ✅ Forward pass successful
# ✅ Training step successful
# ✅ All tests passed!
```

```python
# Cell 7: Test dataset loader
!python scripts/test_ravdess_dataset.py --data_dir /content/drive/MyDrive/RAVDESS

# Expected output:
# ✅ Found 1440 videos
# ✅ 8 emotion classes
# ✅ Train: 1000, Val: 200, Test: 240
```

---

## 🚀 Bước 5: Start Training!

### Option 1: Quick Test (5 epochs)

```python
# Cell 8: Quick test training
!python scripts/train_colab_complete.py \
    --data_dir /content/drive/MyDrive/RAVDESS \
    --save_dir /content/drive/MyDrive/checkpoints/test \
    --config_type lightweight \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --max_epochs 5 \
    --lr 1e-4

# Expected time: ~5-10 minutes
```

### Option 2: Full Training (50 epochs)

```python
# Cell 9: Full training
!python scripts/train_colab_complete.py \
    --data_dir /content/drive/MyDrive/RAVDESS \
    --save_dir /content/drive/MyDrive/checkpoints/full \
    --config_type lightweight \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --max_epochs 50 \
    --lr 1e-4 \
    --early_stopping_patience 15 \
    --save_every 5

# Expected time: ~1.5-2 hours
```

### Option 3: With WandB Logging

```python
# Cell 10: Training with WandB
# First, login to WandB
import wandb
wandb.login()

# Then train
!python scripts/train_colab_complete.py \
    --data_dir /content/drive/MyDrive/RAVDESS \
    --save_dir /content/drive/MyDrive/checkpoints/wandb \
    --config_type lightweight \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --max_epochs 50 \
    --lr 1e-4 \
    --use_wandb \
    --wandb_project multimodal-fer-ravdess
```

---

## 📊 Bước 6: Monitor Training

### 6.1. TensorBoard (Local)

```python
# Cell 11: Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/checkpoints/full/logs
```

### 6.2. WandB (Cloud)

Nếu dùng `--use_wandb`, mở [WandB Dashboard](https://wandb.ai/) để xem:
- Training/validation loss curves
- Accuracy curves
- Learning rate schedule
- System metrics (GPU usage, memory)

### 6.3. Check Training Progress

```python
# Cell 12: Check training history
import json
from pathlib import Path

history_path = Path("/content/drive/MyDrive/checkpoints/full/training_history.json")

if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    
    print(f"Total epochs: {len(history)}")
    print(f"\nLast 5 epochs:")
    for epoch_data in history[-5:]:
        print(f"  Epoch {epoch_data['epoch']}: "
              f"Train Acc={epoch_data['train_acc']:.2f}%, "
              f"Val Acc={epoch_data['val_acc']:.2f}%")
    
    # Best epoch
    best_epoch = max(history, key=lambda x: x['val_acc'])
    print(f"\nBest epoch: {best_epoch['epoch']}")
    print(f"  Val Acc: {best_epoch['val_acc']:.2f}%")
else:
    print("Training history not found. Training may still be in progress.")
```

---

## 🎯 Bước 7: Evaluate Model

```python
# Cell 13: Evaluate on test set
!python scripts/evaluate.py \
    --checkpoint /content/drive/MyDrive/checkpoints/full/best_model.pth \
    --data_dir /content/drive/MyDrive/RAVDESS \
    --batch_size 16

# Expected output:
# Test Accuracy: ~80-85%
# F1-Score: ~0.78-0.82
# Confusion Matrix: ...
```

---

## 💾 Bước 8: Download Checkpoints

```python
# Cell 14: Download best model
from google.colab import files

checkpoint_path = "/content/drive/MyDrive/checkpoints/full/best_model.pth"
files.download(checkpoint_path)

print(f"✅ Downloaded: {checkpoint_path}")
```

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Giải pháp:**
```python
# Giảm batch size
--batch_size 4 \
--grad_accum_steps 4  # Giữ effective batch size = 16
```

### Issue 2: Dataset Not Found

**Giải pháp:**
```python
# Kiểm tra lại đường dẫn
import os
print(os.listdir("/content/drive/MyDrive"))

# Cập nhật đường dẫn
RAVDESS_PATH = "/content/drive/MyDrive/path/to/RAVDESS"
```

### Issue 3: Training Too Slow

**Giải pháp:**
```python
# Dùng lightweight config
--config_type lightweight \
--batch_size 16  # Tăng batch size nếu có đủ VRAM
```

### Issue 4: Disconnected from Runtime

**Giải pháp:**
```python
# Colab Pro có 24h runtime, nhưng có thể bị disconnect
# Để tránh mất progress:

# 1. Save checkpoints thường xuyên
--save_every 5

# 2. Resume training từ checkpoint
!python scripts/train_colab_complete.py \
    --resume_from /content/drive/MyDrive/checkpoints/full/checkpoint_epoch_25.pth \
    --max_epochs 50
```

---

## 📈 Expected Results

### Training Progress

```
Epoch 1/50:
  Train Loss: 1.8234, Acc: 35.23%
  Val Loss: 1.6543, Acc: 42.15%

Epoch 10/50:
  Train Loss: 0.8234, Acc: 68.45%
  Val Loss: 0.9123, Acc: 65.32%

Epoch 25/50:
  Train Loss: 0.4123, Acc: 85.67%
  Val Loss: 0.6234, Acc: 78.45%

Epoch 50/50:
  Train Loss: 0.2345, Acc: 92.34%
  Val Loss: 0.5678, Acc: 82.15%

Best Val Acc: 82.15% (Epoch 50)
Test Acc: 80.45%
```

### Performance by Emotion

```
Emotion Recognition Results:
├─ Neutral: 85%
├─ Calm: 78%
├─ Happy: 88%
├─ Sad: 82%
├─ Angry: 84%
├─ Fearful: 75%
├─ Disgust: 79%
└─ Surprised: 86%

Average: 82%
```

---

## 🎉 Next Steps

### 1. Hyperparameter Tuning

```python
# Try different learning rates
for lr in [1e-5, 5e-5, 1e-4, 5e-4]:
    !python scripts/train_colab_complete.py \
        --lr {lr} \
        --save_dir /content/drive/MyDrive/checkpoints/lr_{lr}
```

### 2. Ablation Studies

```python
# Audio only
!python scripts/train_colab_complete.py --modality audio

# Visual only
!python scripts/train_colab_complete.py --modality visual

# Multimodal (full)
!python scripts/train_colab_complete.py --modality both
```

### 3. Extended Datasets

- CREMA-D (7,442 videos)
- DFEW (16,372 videos)
- MELD (13,708 utterances)

### 4. Model Export

```python
# Export to ONNX
!python scripts/export_onnx.py \
    --checkpoint /content/drive/MyDrive/checkpoints/full/best_model.pth \
    --output model.onnx
```

---

## 📚 Resources

### Documentation
- [ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md) - Kiến trúc chi tiết
- [MODEL_ARCHITECTURE_DIAGRAM.md](MODEL_ARCHITECTURE_DIAGRAM.md) - Sơ đồ trực quan
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Hướng dẫn training chi tiết
- [COLAB_TRAINING_FEASIBILITY.md](COLAB_TRAINING_FEASIBILITY.md) - Phân tích khả thi

### Code
- `models/multimodal_fer.py` - Complete model
- `scripts/train_colab_complete.py` - Training script
- `data/ravdess_dataset.py` - Dataset loader

### Support
- GitHub Issues: [Link to repo]
- Email: [Your email]

---

## ✅ Checklist

Trước khi train, đảm bảo:

- [ ] Google Colab Pro activated
- [ ] RAVDESS dataset uploaded to Google Drive
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] GPU available (A100 40GB)
- [ ] Dataset verified (~1440 videos)
- [ ] Model test passed
- [ ] Training script ready

**Bạn đã sẵn sàng! Chúc may mắn với training! 🚀**

---

## 💡 Pro Tips

1. **Save to Google Drive**: Luôn save checkpoints vào Drive để không mất khi disconnect
2. **Use WandB**: Dễ monitor và compare experiments
3. **Start Small**: Test với 5 epochs trước khi train full
4. **Monitor Memory**: Dùng `nvidia-smi` để check VRAM usage
5. **Backup Checkpoints**: Download checkpoints quan trọng về local

```python
# Monitor GPU usage
!watch -n 1 nvidia-smi
```

Happy Training! 🎉
