# 🚀 Quick Start: Colab via Kiro IDE

Hướng dẫn nhanh để train model trên Colab thông qua Kiro IDE.

## 📋 Prerequisites

1. **Kiro IDE** đã cài đặt và kết nối với Colab
2. **RAVDESS dataset** trong folder `data/ravdess/`
3. **Colab GPU** (T4 Free hoặc A100 Pro)

## 🎯 Option 1: Sử dụng Notebook (Khuyến nghị)

### Bước 1: Mở notebook

```bash
# Trong Kiro IDE, mở file:
train_colab_ide.ipynb
```

### Bước 2: Chạy từng cell

1. **Cell 1**: Check GPU và environment
2. **Cell 2**: Install dependencies
3. **Cell 3**: Verify data
4. **Cell 4**: Import libraries
5. **Cell 5**: Configure training (có thể chỉnh sửa)
6. **Cell 6**: Create model
7. **Cell 7**: Create dataloaders
8. **Cell 8**: Setup training
9. **Cell 9**: Define training functions
10. **Cell 10**: **RUN TRAINING** (2-4 giờ)
11. **Cell 11**: Visualize results
12. **Cell 12**: Test on test set
13. **Cell 13**: Download results

### Bước 3: Monitor training

Training sẽ hiển thị:
- Progress bar cho mỗi epoch
- Train loss realtime
- Validation metrics sau mỗi epoch
- Best UAR được update tự động

### Bước 4: Lấy kết quả

Sau khi training xong:
- Checkpoints: `checkpoints/ravdess_colab/`
- Best model: `checkpoints/ravdess_colab/best_model.pt`
- History: `checkpoints/ravdess_colab/history.json`
- Plots: `checkpoints/ravdess_colab/training_curves.png`

## 🎯 Option 2: Sử dụng Python Script

### Quick test (1 epoch)

```bash
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --epochs 1 \
    --batch_size 4 \
    --save_dir checkpoints/test
```

### Full training

```bash
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --epochs 100 \
    --batch_size 16 \
    --save_dir checkpoints/ravdess_full
```

### Custom configuration

```bash
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --modality speech \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --num_audio_layers 8 \
    --num_visual_layers 4 \
    --num_fusion_layers 4 \
    --save_dir checkpoints/my_experiment
```

## 🔧 Configuration Tips

### Cho T4 GPU (Free Colab)

```python
CONFIG = {
    "batch_size": 8,           # Nhỏ hơn để tránh OOM
    "num_audio_layers": 8,
    "num_visual_layers": 4,
    "num_fusion_layers": 4,
    "use_pretrained_visual": False,
    "use_pretrained_fusion": False,
}
```

### Cho A100 GPU (Colab Pro)

```python
CONFIG = {
    "batch_size": 32,          # Lớn hơn cho training nhanh
    "num_audio_layers": 12,
    "num_visual_layers": 6,
    "num_fusion_layers": 6,
    "use_pretrained_visual": True,   # Có thể dùng pretrained
    "use_pretrained_fusion": False,  # Vẫn nên False (có bug)
}
```

## 📊 Expected Results

### Dataset Split
- **Train**: 1920 videos (actors 1-16)
- **Val**: 480 videos (actors 17-20)
- **Test**: 480 videos (actors 21-24)

### Performance (sau 100 epochs)
- **UAR**: 0.65-0.75
- **Accuracy**: 0.70-0.80
- **Training time**: 
  - T4: 3-4 giờ
  - A100: 1-2 giờ

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"

**Giải pháp:**
```python
# Giảm batch size
CONFIG["batch_size"] = 4

# Hoặc giảm model size
CONFIG["num_audio_layers"] = 6
CONFIG["num_visual_layers"] = 3
CONFIG["num_fusion_layers"] = 3
```

### Lỗi: "Loaded 0 videos"

**Kiểm tra:**
```python
from pathlib import Path

data_path = Path("data/ravdess")
print(f"Exists: {data_path.exists()}")

folders = list(data_path.glob("Video_Speech_Actor_*"))
print(f"Actors: {len(folders)}")
```

**Giải pháp:** Đảm bảo data structure:
```
data/ravdess/
├── Video_Speech_Actor_01/
│   └── Actor_01/
│       ├── 01-01-01-01-01-01-01.mp4
│       └── ...
├── Video_Speech_Actor_02/
│   └── Actor_02/
│       └── ...
└── ...
```

### Lỗi: "ffmpeg not found"

**Colab:**
```bash
!apt-get install -y ffmpeg
```

**Local:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Training quá chậm

**Tăng tốc:**
1. Tăng batch size (nếu có đủ VRAM)
2. Giảm num_workers nếu I/O chậm
3. Sử dụng mixed precision (đã bật mặc định)
4. Giảm số layers nếu không cần accuracy cao

## 📈 Monitoring Progress

### Trong notebook

Training sẽ tự động hiển thị:
- Progress bar với loss realtime
- Validation metrics sau mỗi epoch
- Best model được save tự động

### Check history

```python
import json

with open("checkpoints/ravdess_colab/history.json") as f:
    history = json.load(f)

print(f"Epochs: {len(history['train_loss'])}")
print(f"Best UAR: {max(history['val_uar']):.4f}")
print(f"Latest loss: {history['train_loss'][-1]:.4f}")
```

### Visualize

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_uar'])
plt.xlabel('Epoch')
plt.ylabel('UAR')
plt.title('Validation UAR')

plt.tight_layout()
plt.show()
```

## 💾 Save & Download Results

### Trong Colab notebook

Chạy cell cuối để zip và download:
```python
!zip -r results.zip checkpoints/ravdess_colab
```

### Qua IDE

Files sẽ tự động sync về local machine qua Kiro IDE.

## 🎓 Next Steps

Sau khi training xong:

1. **Evaluate**: Test trên test set (cell 12)
2. **Inference**: Sử dụng `scripts/inference.py`
3. **Fine-tune**: Adjust hyperparameters và train lại
4. **Deploy**: Export model cho production

## 📚 Related Files

- `train_colab_ide.ipynb` - Main training notebook
- `scripts/train_ravdess.py` - Training script
- `scripts/test_local_setup.py` - Setup verification
- `TRAIN_LOCAL_GUIDE.md` - Detailed training guide
- `data/ravdess_dataset.py` - Dataset loader

## 🆘 Need Help?

1. Check `TEST_STATUS.md` for known issues
2. Run `python scripts/test_local_setup.py` to verify setup
3. Check logs in `checkpoints/*/history.json`
4. Review configuration in `checkpoints/*/config.json`
