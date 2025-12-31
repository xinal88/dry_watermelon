# 📦 Tóm Tắt Setup Colab - Quick Reference

## 🎯 Mục Tiêu
Train Multimodal FER model trên Google Colab Pro với full RAVDESS dataset.

---

## ✅ Đã Chuẩn Bị

### 1. **Dataset Loader** ✅
- File: `data/ravdess_dataset.py`
- Hỗ trợ: Speech và Song modality
- Train/Val/Test split: Actors 1-16 / 17-20 / 21-24
- Audio extraction với ffmpeg
- Video frame extraction

### 2. **Training Guide** ✅
- File: `COLAB_TRAINING_GUIDE.md`
- Hướng dẫn chi tiết từng bước
- Code cells sẵn sàng copy-paste
- Configuration đầy đủ

### 3. **Requirements** ✅
- File: `requirements_colab.txt`
- Tất cả dependencies cần thiết
- Tương thích với Colab

### 4. **Test Script** ✅
- File: `scripts/test_ravdess_dataset.py`
- Test dataset loader trước khi train
- Kiểm tra data distribution

---

## 🚀 Quick Start (3 Bước)

### Bước 1: Chuẩn Bị Drive

```
My Drive/
└── RAVDESS_Multimodal_FER/
    ├── data/
    │   └── ravdess/
    │       ├── Video_Speech_Actor_01/
    │       ├── ...
    │       └── Video_Speech_Actor_24/
    ├── models/
    ├── training/
    ├── scripts/
    └── requirements_colab.txt
```

**Upload code:**
- Nén project: `dry_watermelon.zip`
- Upload lên Drive
- Giải nén trên Colab

### Bước 2: Tạo Colab Notebook

1. Mở Google Colab: https://colab.research.google.com
2. New Notebook
3. Runtime → Change runtime type → GPU (T4 hoặc A100)
4. Copy code từ `COLAB_TRAINING_GUIDE.md`

### Bước 3: Run Training

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/RAVDESS_Multimodal_FER

# Cell 3: Install
!pip install -r requirements_colab.txt

# Cell 4-13: Copy từ COLAB_TRAINING_GUIDE.md
```

---

## 📊 Cấu Hình Khuyến Nghị

### Lightweight (Nhanh, 2-3 giờ):
```python
CONFIG = {
    "batch_size": 16,
    "num_epochs": 100,
    "audio_dim": 512,
    "visual_dim": 512,
    "num_audio_layers": 8,
    "num_visual_layers": 4,
    "num_fusion_layers": 4,
    "use_pretrained_visual": False,
    "use_pretrained_fusion": False,
}
# Expected UAR: 75-80%
```

### Full Pretrained (Chậm hơn, 4-6 giờ):
```python
CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "audio_dim": 512,
    "visual_dim": 768,
    "num_audio_layers": 17,
    "num_visual_layers": 6,
    "num_fusion_layers": 6,
    "use_pretrained_visual": True,  # SigLIP2
    "use_pretrained_fusion": True,  # LFM2-700M
}
# Expected UAR: 80-85%
```

---

## 🔍 Test Trước Khi Train

### Test trên Local (IDE):
```bash
# Test dataset loader
python scripts/test_ravdess_dataset.py

# Test model forward pass
python scripts/quick_test.py
```

### Test trên Colab:
```python
# Trong notebook, sau khi mount Drive
!python scripts/test_ravdess_dataset.py
```

---

## 📁 Files Quan Trọng

| File | Mục Đích |
|------|----------|
| `data/ravdess_dataset.py` | Dataset loader cho RAVDESS |
| `COLAB_TRAINING_GUIDE.md` | Hướng dẫn chi tiết train trên Colab |
| `requirements_colab.txt` | Dependencies cho Colab |
| `scripts/test_ravdess_dataset.py` | Test dataset loader |
| `scripts/inference_cpu.py` | Inference trên local sau khi train |

---

## 💾 Sau Khi Train Xong

### 1. Download Checkpoints:
```python
# Trong Colab
from google.colab import files
files.download("checkpoints/ravdess_speech/best_model.pth")
files.download("checkpoints/ravdess_speech/training_history.json")
```

### 2. Test trên Local:
```bash
# Copy best_model.pth về local
# Sửa CONFIG trong scripts/inference_cpu.py:
CONFIG = {
    "checkpoint_path": "checkpoints/ravdess_speech/best_model.pth",
    "video_path": "data/test_samples/01-02-01-01-01-01-01.mp4",
    ...
}

# Run inference
python scripts/inference_cpu.py
```

### 3. Evaluate trên Test Set:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ravdess_speech/best_model.pth \
    --data-dir data/ravdess \
    --split test
```

---

## 🎯 Workflow Hoàn Chỉnh

```
1. Local (IDE) - Development & Testing
   ├── Viết code
   ├── Test với test_samples (3 videos)
   ├── Debug và fix bugs
   └── Verify model architecture

2. Colab Pro - Training
   ├── Upload code + data lên Drive
   ├── Train với full RAVDESS dataset
   ├── Monitor training curves
   └── Download best checkpoint

3. Local (IDE) - Inference & Deployment
   ├── Load trained checkpoint
   ├── Test inference
   ├── Evaluate performance
   └── Deploy model
```

---

## 🐛 Troubleshooting

### Issue 1: "No module named 'models'"
```python
# Thêm vào đầu notebook
import sys
sys.path.insert(0, "/content/drive/MyDrive/RAVDESS_Multimodal_FER")
```

### Issue 2: "CUDA out of memory"
```python
# Giảm batch size
CONFIG["batch_size"] = 8  # hoặc 4

# Hoặc dùng gradient accumulation
CONFIG["gradient_accumulation_steps"] = 4
```

### Issue 3: "ffmpeg not found"
```python
# Colab đã có ffmpeg, nhưng nếu lỗi:
CONFIG["use_audio"] = False  # Tạm thời không dùng audio
```

### Issue 4: Training quá chậm
```python
# Giảm model size
CONFIG["num_audio_layers"] = 4
CONFIG["num_visual_layers"] = 2
CONFIG["num_fusion_layers"] = 2

# Hoặc dùng ít epochs
CONFIG["num_epochs"] = 50
```

---

## 📈 Kết Quả Mong Đợi

### Dataset Size:
- **Train**: ~960 videos (Actors 1-16)
- **Val**: ~240 videos (Actors 17-20)
- **Test**: ~240 videos (Actors 21-24)
- **Total**: ~1440 videos

### Performance:
- **Lightweight**: 75-80% UAR
- **Full Pretrained**: 80-85% UAR
- **Training time**: 2-6 giờ (tùy config)

### Checkpoints:
- `best_model.pth`: Model tốt nhất (theo UAR)
- `final_model.pth`: Model cuối cùng
- `training_history.json`: Lịch sử training
- `training_curves.png`: Đồ thị training

---

## ✅ Checklist Cuối Cùng

**Trước khi train:**
- [ ] Dữ liệu RAVDESS đã upload lên Drive
- [ ] Code đã upload lên Drive
- [ ] Đã test dataset loader
- [ ] Đã chọn GPU runtime trên Colab
- [ ] Đã mount Drive thành công

**Trong khi train:**
- [ ] Monitor training curves
- [ ] Check UAR tăng dần
- [ ] Không có OOM errors
- [ ] Checkpoints được lưu đều đặn

**Sau khi train:**
- [ ] Download best_model.pth
- [ ] Download training_history.json
- [ ] Test inference trên local
- [ ] Evaluate trên test set

---

## 🎉 Hoàn Thành!

Bạn đã có đầy đủ:
1. ✅ Dataset loader cho RAVDESS
2. ✅ Hướng dẫn train chi tiết
3. ✅ Code sẵn sàng cho Colab
4. ✅ Scripts test và inference

**Sẵn sàng train trên Colab Pro!** 🚀

---

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra `COLAB_TRAINING_GUIDE.md` - Hướng dẫn chi tiết
2. Test với `scripts/test_ravdess_dataset.py`
3. Xem phần Troubleshooting ở trên
4. Check GPU memory với `!nvidia-smi`

**Good luck with training!** 🎯
