# ✅ SẴN SÀNG TRAINING TRÊN COLAB IDE

## 📦 Files Đã Tạo

### 1. Notebook Training
- **File**: `Train_Multimodal_FER.ipynb`
- **Mô tả**: Jupyter notebook với 30 cells để training full RAVDESS
- **Sử dụng**: Mở trong IDE với Colab extension

### 2. Hướng Dẫn Chi Tiết
- **File**: `COLAB_IDE_SETUP.md`
- **Mô tả**: Hướng dẫn đầy đủ từng bước setup và training
- **Nội dung**: Upload data, config, troubleshooting

### 3. Quick Start
- **File**: `QUICK_START_COLAB.md`
- **Mô tả**: Hướng dẫn nhanh 3 bước
- **Sử dụng**: Cho người muốn bắt đầu ngay

---

## 🚀 Bắt Đầu Ngay

### Bước 1: Upload RAVDESS lên Google Drive

```
My Drive/
└── RAVDESS/
    ├── Actor_01/
    ├── Actor_02/
    └── ... (24 actors)
```

**Thời gian**: 30-60 phút (tùy tốc độ mạng)

### Bước 2: Mở Notebook

1. Mở file `Train_Multimodal_FER.ipynb` trong IDE
2. Connect to Google Colab kernel
3. Chọn GPU runtime (T4 hoặc A100)

### Bước 3: Chạy Training

- Click "Run All" hoặc chạy từng cell
- Đợi 2-4 giờ
- Download checkpoints khi xong

---

## ⚙️ Cấu Hình Khuyến Nghị

### Cho T4 GPU (Free/Pro):

```python
CONFIG = {
    "batch_size": 16,
    "num_epochs": 100,
    "num_audio_layers": 8,
    "num_visual_layers": 4,
    "num_fusion_layers": 4,
    "use_pretrained_visual": False,
    "use_pretrained_fusion": False,
}
```

**Kết quả dự kiến**:
- UAR: 75-80%
- Thời gian: 2-3 giờ
- Model size: ~150M params

### Cho A100 GPU (Pro+):

```python
CONFIG = {
    "batch_size": 32,
    "num_epochs": 100,
    "num_audio_layers": 17,
    "num_visual_layers": 6,
    "num_fusion_layers": 6,
    "use_pretrained_visual": True,
    "use_pretrained_fusion": True,
}
```

**Kết quả dự kiến**:
- UAR: 80-85%
- Thời gian: 4-6 giờ
- Model size: ~393M params

---

## 📊 Notebook Structure

### 30 Cells tổng cộng:

1. **Cell 1**: Check GPU ✓
2. **Cell 2**: Clone repo (nếu dùng GitHub)
3. **Cell 3**: Mount Google Drive ✓
4. **Cell 4**: Install dependencies ✓
5. **Cell 5**: Import libraries ✓
6. **Cell 6**: Configuration ⚙️ (EDIT THIS)
7. **Cell 7**: Create model ✓
8. **Cell 8**: Create dataloaders ✓
9. **Cell 9**: Training setup ✓
10. **Cell 10**: Training functions ✓
11. **Cell 11**: Main training loop 🚀 (2-4 hours)
12. **Cell 12**: Plot training curves 📈
13. **Cell 13**: Evaluate on test set 🧪
14. **Cell 14**: Download checkpoints 💾

---

## 🎯 Workflow

```
┌─────────────────────────────────────────┐
│ 1. Upload RAVDESS to Google Drive      │
│    (30-60 mins, one-time)               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 2. Open Train_Multimodal_FER.ipynb     │
│    in IDE with Colab extension          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 3. Edit CONFIG in Cell 6                │
│    (batch_size, pretrained, etc.)       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 4. Run All Cells                        │
│    (2-4 hours training)                 │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 5. Download Checkpoints                 │
│    (best_model.pth, history, etc.)      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 6. Test on Local with inference_cpu.py │
└─────────────────────────────────────────┘
```

---

## 💾 Checkpoints Sau Training

Sau khi training xong, bạn sẽ có:

```
checkpoints/ravdess_speech_t4/
├── best_model.pth              # Model tốt nhất (theo UAR)
├── final_model.pth             # Model epoch cuối
├── checkpoint_epoch_10.pth     # Checkpoint định kỳ
├── checkpoint_epoch_20.pth
├── ...
├── training_history.json       # Lịch sử training
├── test_results.json           # Kết quả test set
└── training_curves.png         # Đồ thị metrics
```

---

## 🧪 Test Model

Sau khi download checkpoints về local:

```bash
# 1. Copy checkpoint vào project
mkdir -p checkpoints/ravdess_speech_t4
mv ~/Downloads/best_model.pth checkpoints/ravdess_speech_t4/

# 2. Chỉnh sửa inference script
# File: scripts/inference_cpu.py
# Line: CHECKPOINT_PATH = "checkpoints/ravdess_speech_t4/best_model.pth"

# 3. Chạy inference
python scripts/inference_cpu.py
```

---

## 📈 Metrics Quan Trọng

### UAR (Unweighted Average Recall) ⭐
- **Metric chính** cho emotion recognition
- Target: >75% (lightweight) hoặc >80% (pretrained)
- Đo lường khả năng nhận diện đều các emotions

### Accuracy
- Độ chính xác tổng thể
- Target: >78% (lightweight) hoặc >83% (pretrained)

### Loss
- Giảm dần theo epochs
- Train loss < Val loss = normal
- Val loss tăng = overfitting

---

## ⚠️ Troubleshooting

### Lỗi OOM (Out of Memory)

```python
# Giảm batch size
"batch_size": 8,  # từ 16

# Hoặc tăng gradient accumulation
"gradient_accumulation_steps": 2,  # từ 1
```

### RAVDESS không tìm thấy

```python
# Kiểm tra đường dẫn
!ls /content/drive/MyDrive/RAVDESS

# Hoặc thay đổi path
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"
```

### Colab disconnect

- Training tự động save checkpoint mỗi 10 epochs
- Chạy lại từ Cell 11 (Main Training Loop)
- Model sẽ resume từ checkpoint cuối

---

## 📞 Cần Trợ Giúp?

### Đọc tài liệu:
1. `COLAB_IDE_SETUP.md` - Hướng dẫn chi tiết
2. `QUICK_START_COLAB.md` - Quick start
3. `COLAB_TRAINING_GUIDE.md` - Training guide gốc

### Kiểm tra:
- GPU: `!nvidia-smi`
- RAVDESS: `!ls data/ravdess | head -20`
- Logs: Xem output của từng cell

---

## 🎉 Sẵn Sàng!

Bạn đã có mọi thứ cần thiết để training:

✅ Notebook với 30 cells  
✅ Hướng dẫn chi tiết  
✅ Cấu hình tối ưu  
✅ Troubleshooting guide  

**Bắt đầu training ngay!** 🚀

---

## 📝 Checklist

- [ ] Upload RAVDESS lên Google Drive
- [ ] Mở `Train_Multimodal_FER.ipynb` trong IDE
- [ ] Connect to Colab kernel (T4 hoặc A100)
- [ ] Edit CONFIG trong Cell 6
- [ ] Run All Cells
- [ ] Theo dõi training progress (2-4 giờ)
- [ ] Download checkpoints
- [ ] Test với `inference_cpu.py`

**Good luck! 🍀**
