# 🚀 Hướng Dẫn Training trên Colab với IDE Extension

## 📋 Yêu Cầu

✅ Đã cài Colab extension trong IDE  
✅ Có tài khoản Google Colab Pro (T4 GPU)  
✅ Đã upload RAVDESS dataset lên Google Drive

---

## 📂 Bước 1: Upload RAVDESS lên Google Drive

### Cấu trúc thư mục trên Drive:

```
My Drive/
└── RAVDESS/
    ├── Actor_01/
    │   ├── 01-01-01-01-01-01-01.mp4
    │   ├── 01-01-01-01-01-02-01.mp4
    │   └── ...
    ├── Actor_02/
    ├── Actor_03/
    └── ...
    └── Actor_24/
```

### Cách upload:

1. Mở Google Drive: https://drive.google.com
2. Tạo folder `RAVDESS` trong `My Drive`
3. Upload tất cả folders `Actor_01` đến `Actor_24` vào folder `RAVDESS`
4. Đợi upload hoàn tất (có thể mất 30-60 phút tùy tốc độ mạng)

---

## 🎯 Bước 2: Mở Notebook trong IDE

1. Mở file `Train_Multimodal_FER.ipynb` trong IDE
2. IDE sẽ tự động nhận diện đây là Colab notebook
3. Click vào nút "Connect to Colab" hoặc chọn kernel "Google Colab"

---

## ⚙️ Bước 3: Cấu Hình Training

Trong notebook, tìm đến **Step 6: Configuration** và chỉnh sửa:

```python
CONFIG = {
    # ============ DATA ============
    "data_dir": "data/ravdess",
    "modality": "speech",      # "speech" hoặc "song"
    
    # ============ TRAINING ============
    "batch_size": 16,          # T4: 8-16, A100: 32-64
    "num_epochs": 100,
    "lr": 1e-4,
    
    # ============ MODEL ============
    # Lightweight cho T4 GPU (4GB VRAM)
    "num_audio_layers": 8,     # Giảm từ 17
    "num_visual_layers": 4,    # Giảm từ 6
    "num_fusion_layers": 4,    # Giảm từ 6
    
    # ============ PRETRAINED ============
    "use_pretrained_visual": False,  # True = tốt hơn nhưng chậm hơn
    "use_pretrained_fusion": False,  # True = tốt hơn nhưng chậm hơn
    
    # ============ CHECKPOINTING ============
    "save_dir": "checkpoints/ravdess_speech_t4",
}
```

### Lựa chọn cấu hình:

#### Option A: Lightweight (Khuyến nghị cho T4)
- `use_pretrained_visual`: False
- `use_pretrained_fusion`: False
- `num_audio_layers`: 8
- `num_visual_layers`: 4
- `num_fusion_layers`: 4
- **Thời gian**: 2-3 giờ
- **UAR dự kiến**: 75-80%

#### Option B: Full Pretrained (Cho A100)
- `use_pretrained_visual`: True
- `use_pretrained_fusion`: True
- `num_audio_layers`: 17
- `num_visual_layers`: 6
- `num_fusion_layers`: 6
- **Thời gian**: 4-6 giờ
- **UAR dự kiến**: 80-85%

---

## 🚀 Bước 4: Chạy Training

### Chạy từng cell theo thứ tự:

1. **Cell 1**: Check GPU
   - Xác nhận có T4 GPU
   - VRAM: ~15GB

2. **Cell 2**: Mount Google Drive
   - Cho phép truy cập Drive
   - Xác nhận đường dẫn RAVDESS đúng

3. **Cell 3**: Install Dependencies
   - Cài đặt thư viện cần thiết
   - Mất ~2-3 phút

4. **Cell 4-6**: Import & Config
   - Import libraries
   - Kiểm tra cấu hình

5. **Cell 7-8**: Create Model & Data
   - Tạo model (~150M params)
   - Load RAVDESS dataset

6. **Cell 9-11**: Training
   - **Đây là bước chính - mất 2-4 giờ**
   - Theo dõi metrics: UAR, Loss
   - Model tự động save best checkpoint

7. **Cell 12-13**: Evaluation & Plots
   - Đánh giá trên test set
   - Vẽ training curves

8. **Cell 14**: Download Checkpoints
   - Download về máy local

---

## 📊 Theo Dõi Training

### Metrics quan trọng:

- **UAR** (Unweighted Average Recall): Metric chính ⭐
  - Target: >75% (lightweight) hoặc >80% (pretrained)
- **Loss**: Giảm dần theo epochs
- **Accuracy**: Độ chính xác tổng thể

### Training progress:

```
Epoch 1/100
  Train Loss: 1.8234
  Val Loss:   1.7123
  UAR:        0.3456 ⭐
  
Epoch 10/100
  Train Loss: 0.9234
  Val Loss:   1.0123
  UAR:        0.6234 ⭐
  🎉 New best UAR: 0.6234
  
...

Epoch 100/100
  Train Loss: 0.2134
  Val Loss:   0.4523
  UAR:        0.7823 ⭐
```

---

## 💾 Bước 5: Download Checkpoints

Sau khi training xong, notebook sẽ tự động download:

1. `best_model.pth` - Model tốt nhất (theo UAR)
2. `training_history.json` - Lịch sử training
3. `test_results.json` - Kết quả test set
4. `training_curves.png` - Đồ thị training

### Lưu checkpoints vào project:

```bash
# Copy vào thư mục checkpoints
mkdir -p checkpoints/ravdess_speech_t4
mv best_model.pth checkpoints/ravdess_speech_t4/
mv training_history.json checkpoints/ravdess_speech_t4/
mv test_results.json checkpoints/ravdess_speech_t4/
mv training_curves.png checkpoints/ravdess_speech_t4/
```

---

## 🧪 Bước 6: Test Model trên Local

Sau khi download checkpoints, test trên máy local:

```bash
# Chỉnh sửa scripts/inference_cpu.py
# Thay đổi checkpoint path:
CHECKPOINT_PATH = "checkpoints/ravdess_speech_t4/best_model.pth"

# Chạy inference
python scripts/inference_cpu.py
```

---

## ⚠️ Xử Lý Lỗi

### Lỗi 1: Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Giải pháp**:
- Giảm `batch_size` từ 16 → 8
- Tăng `gradient_accumulation_steps` từ 1 → 2
- Giảm số layers trong model

### Lỗi 2: RAVDESS không tìm thấy

```
FileNotFoundError: data/ravdess not found
```

**Giải pháp**:
- Kiểm tra đường dẫn Drive: `/content/drive/MyDrive/RAVDESS`
- Đảm bảo đã mount Drive thành công
- Kiểm tra symlink: `!ls -la data/ravdess`

### Lỗi 3: Colab disconnect

**Giải pháp**:
- Training sẽ tự động save checkpoint mỗi 10 epochs
- Nếu disconnect, chạy lại từ cell "Main Training Loop"
- Model sẽ load từ checkpoint cuối cùng

---

## 📈 Kết Quả Mong Đợi

### Lightweight Model (T4):
- **Parameters**: ~150M
- **Training time**: 2-3 giờ
- **UAR**: 75-80%
- **Accuracy**: 78-83%

### Full Pretrained (A100):
- **Parameters**: ~393M
- **Training time**: 4-6 giờ
- **UAR**: 80-85%
- **Accuracy**: 83-88%

---

## 🎉 Hoàn Thành!

Sau khi training xong, bạn có:

✅ Trained model với UAR >75%  
✅ Checkpoints để inference  
✅ Training curves và metrics  
✅ Test results trên RAVDESS

### Next Steps:

1. Test model trên video mới với `inference_cpu.py`
2. Fine-tune với hyperparameters khác nếu cần
3. Deploy model cho production

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:

1. Kiểm tra GPU: `!nvidia-smi`
2. Kiểm tra RAVDESS: `!ls data/ravdess | head -20`
3. Kiểm tra logs trong notebook
4. Giảm batch_size nếu OOM

**Good luck với training! 🚀**
