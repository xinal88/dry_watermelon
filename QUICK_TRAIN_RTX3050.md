# 🚀 Quick Training Guide - RTX 3050

Train nhanh với 50% dataset trên RTX 3050 (4GB VRAM).

## ⚡ Quick Start

```bash
# Chạy training ngay
python scripts/train_half_dataset.py
```

Đơn giản vậy thôi! Script sẽ tự động:
- Dùng 50% training data (960 samples thay vì 1920)
- Model nhẹ hơn (256 dim thay vì 512)
- Batch size nhỏ (4 thay vì 8-16)
- 50 epochs (thay vì 100)

## 📊 Thông số tối ưu cho RTX 3050

### Model Architecture (Lightweight)
```
Audio Branch:    256 dim, 6 layers  (was 512 dim, 8 layers)
Visual Branch:   256 dim, 3 layers  (was 512 dim, 4 layers)
Fusion:          512 dim, 3 layers  (was 1024 dim, 4 layers)
Total params:    ~50M               (was ~150M)
```

### Training Config
```
Batch size:      4
Epochs:          50
Learning rate:   1e-4
Mixed precision: FP16 (enabled)
Gradient clip:   1.0
```

### Dataset Split (50% training data)
```
Train:  960 samples  (actors 1-16, random 50%)
Val:    480 samples  (actors 17-20, full)
Test:   480 samples  (actors 21-24, full)
```

## ⏱️ Expected Performance

### Training Time
- **Per epoch**: ~2-3 minutes
- **Total (50 epochs)**: ~1.5-2 hours
- **VRAM usage**: ~3.5 GB (safe for 4GB)

### Expected Metrics (after 50 epochs)
- **UAR**: 0.55-0.65 (lower than full dataset, but acceptable)
- **Accuracy**: 0.60-0.70
- **Val UAR**: Should improve steadily

## 📁 Output

Training sẽ tạo folder:
```
checkpoints/half_dataset_rtx3050/
├── config.json              # Training configuration
├── history.json             # Training history
├── best_model.pt            # Best model (highest val UAR)
├── checkpoint_epoch_10.pt   # Checkpoint at epoch 10
├── checkpoint_epoch_20.pt   # Checkpoint at epoch 20
├── checkpoint_epoch_30.pt   # Checkpoint at epoch 30
├── checkpoint_epoch_40.pt   # Checkpoint at epoch 40
├── checkpoint_epoch_50.pt   # Checkpoint at epoch 50
└── test_results.json        # Final test results
```

## 📈 Monitoring Progress

Script sẽ hiển thị realtime:

```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|████████████| 240/240 [02:15<00:00, loss=2.0543]
Validation: 100%|██████████| 120/120 [00:30<00:00]

Results:
  Train Loss: 2.0543
  Val Loss:   1.8234
  Accuracy:   0.3542
  UAR:        0.3125
  Time:       165.3s
  ETA: 123.8 minutes
```

## 🎯 Sau khi training xong

### 1. Check results
```python
import json

# Load history
with open("checkpoints/half_dataset_rtx3050/history.json") as f:
    history = json.load(f)

print(f"Best UAR: {max(history['val_uar']):.4f}")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
```

### 2. Visualize training
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
plt.savefig('training_curves.png')
plt.show()
```

### 3. Test inference
```python
import torch
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig

# Load best model
checkpoint = torch.load("checkpoints/half_dataset_rtx3050/best_model.pt")

# Create model (same config as training)
audio_config = AudioBranchConfig(feature_dim=256, num_layers=6)
visual_config = VisualBranchConfig(feature_dim=256, temporal_depth=3)
fusion_config = LFM2FusionConfig(
    num_layers=3, hidden_dim=512,
    audio_dim=256, visual_dim=256, output_dim=256
)

model = MultimodalFER(
    audio_config=audio_config,
    visual_config=visual_config,
    fusion_config=fusion_config,
    num_classes=8,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
```

## 🔧 Troubleshooting

### Nếu vẫn bị OOM (Out of Memory)

Giảm batch size:
```python
# Edit line 202 in scripts/train_half_dataset.py
"batch_size": 2,  # Change from 4 to 2
```

### Nếu muốn train nhanh hơn

Giảm epochs:
```python
# Edit line 203 in scripts/train_half_dataset.py
"num_epochs": 30,  # Change from 50 to 30
```

### Nếu muốn dùng full dataset

Sử dụng script gốc:
```bash
python scripts/train_ravdess.py \
    --data_dir data/ravdess \
    --batch_size 4 \
    --epochs 50 \
    --audio_dim 256 \
    --visual_dim 256 \
    --fusion_hidden_dim 512 \
    --num_audio_layers 6 \
    --num_visual_layers 3 \
    --num_fusion_layers 3
```

## 💡 Tips

### Tăng performance
1. **Close other apps** để giải phóng VRAM
2. **Disable browser** nếu có GPU acceleration
3. **Set num_workers=0** nếu CPU yếu

### Monitor GPU
```bash
# Trong terminal khác
watch -n 1 nvidia-smi
```

### Save VRAM
```python
# Nếu cần, có thể giảm thêm:
- audio_dim: 256 -> 128
- visual_dim: 256 -> 128
- fusion_hidden_dim: 512 -> 256
```

## 📊 So sánh với Full Training

| Metric | Half Dataset (50 epochs) | Full Dataset (100 epochs) |
|--------|--------------------------|---------------------------|
| Training samples | 960 | 1920 |
| Training time | 1.5-2 hours | 3-4 hours |
| Expected UAR | 0.55-0.65 | 0.65-0.75 |
| VRAM usage | ~3.5 GB | ~3.8 GB |
| Model size | ~50M params | ~150M params |

## ✅ Advantages

- ✅ **Nhanh**: 1.5-2 giờ thay vì 3-4 giờ
- ✅ **An toàn**: Chắc chắn không OOM
- ✅ **Đủ dùng**: UAR 0.55-0.65 vẫn acceptable
- ✅ **Đơn giản**: Chỉ 1 command

## 🎓 Next Steps

Sau khi train xong:

1. **Evaluate**: Check test results trong `test_results.json`
2. **Inference**: Dùng model cho prediction
3. **Fine-tune**: Nếu cần, train thêm với learning rate nhỏ hơn
4. **Full training**: Nếu kết quả tốt, có thể train full dataset sau

## 🆘 Need Help?

- Check `history.json` để xem training progress
- Check `config.json` để xem configuration
- Run `python scripts/test_local_setup.py` để verify setup
- Check GPU usage: `nvidia-smi`

---

**Ready to train?**
```bash
python scripts/train_half_dataset.py
```

Ngồi uống cà phê và chờ 1.5-2 giờ! ☕
