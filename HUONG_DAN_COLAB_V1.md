# 🎯 Hướng Dẫn Sử Dụng train_dry_watermelon_v1.ipynb

## ✅ Các Cải Tiến So Với Phiên Bản Cũ

### Vấn Đề Đã Fix:
1. ✅ **Lỗi "Loaded 0 videos"** - Đã fix dataset loader hỗ trợ cả `Actor_*` và `Video_Speech_Actor_*`
2. ✅ **Mount Drive trong Colab IDE** - Tự động detect và mount đúng cách
3. ✅ **Validation dữ liệu** - Kiểm tra dữ liệu TRƯỚC KHI tạo model
4. ✅ **Error handling tốt hơn** - Thông báo lỗi rõ ràng, dễ debug
5. ✅ **Tương thích cả Colab web và Colab IDE extension**

### Các Tính Năng Mới:
- 🔍 Auto-detect môi trường Colab
- 📊 Validation dữ liệu chi tiết
- 💾 Tự động save checkpoints
- 📈 Visualization training curves
- 🎯 Test set evaluation
- 📦 Easy download checkpoints

## 🚀 Cách Sử Dụng

### Bước 1: Push Code Lên GitHub

```bash
# Add files
git add data/ravdess_dataset.py
git add train_dry_watermelon_v1.ipynb
git add HUONG_DAN_COLAB_V1.md

# Commit
git commit -m "Add v1 notebook with fixes for Colab IDE"

# Push
git push origin main
```

### Bước 2: Mở Notebook Trong Colab

**Option A: Colab Web (Khuyến nghị)**
1. Vào https://colab.research.google.com/
2. File → Open notebook → GitHub
3. Nhập: `xinal88/dry_watermelon`
4. Chọn: `train_dry_watermelon_v1.ipynb`

**Option B: Colab IDE Extension (VS Code)**
1. Mở VS Code
2. Cài extension: "Colab"
3. Open file: `train_dry_watermelon_v1.ipynb`
4. Click "Open in Colab"

### Bước 3: Chọn GPU Runtime

1. Click `Runtime` → `Change runtime type`
2. Chọn `T4 GPU` (miễn phí) hoặc `A100` (Colab Pro)
3. Click `Save`

### Bước 4: Chạy Từng Cell

**QUAN TRỌNG**: Chạy theo thứ tự, KHÔNG skip cell nào!

#### Cell 1: Environment Check ✅
```python
# Kiểm tra môi trường
Running in Colab: True
✓ Colab environment detected
```

#### Cell 2: Clone Repository ✅
```python
# Clone code từ GitHub
Cloning repository...
✓ Repository cloned
```

#### Cell 3: Mount Google Drive ⚠️ QUAN TRỌNG!
```python
# CẬP NHẬT ĐƯỜNG DẪN NÀY!
RAVDESS_PATH = "/content/drive/MyDrive/[HUST]_Facial_Expression_Recognition/Dataset/Multimodal_DFER/RAVDESS"
```

**Nếu đường dẫn của bạn khác, sửa lại cho đúng!**

Kết quả mong đợi:
```
✓ Google Drive already mounted
✓ RAVDESS path found: /content/drive/MyDrive/...
✓ Found 24 Actor folders
  Sample: ['Actor_01', 'Actor_02', 'Actor_03']
✓ Created symlink: data/ravdess -> Google Drive
```

#### Cell 4: Install Dependencies ✅
```python
# Cài đặt packages
Installing dependencies...
✓ All dependencies installed!
```

#### Cell 5: Import Libraries ✅
```python
# Import modules
✓ All imports successful!
```

#### Cell 6: Configuration ⚙️
```python
# Xem cấu hình training
TRAINING CONFIGURATION
======================================================================
  data_dir: /content/drive/MyDrive/.../RAVDESS
  batch_size: 16
  num_epochs: 40
  ...
```

**Có thể chỉnh sửa:**
- `batch_size`: 8-16 cho T4, 32-64 cho A100
- `num_epochs`: Số epoch (40 = ~2-3 giờ)
- `use_audio`: True/False (có dùng audio không)

#### Cell 7: Validate Data ✅ QUAN TRỌNG!
```python
# Kiểm tra dữ liệu
Found 24 video folders:
  Sample folders: ['Actor_01', 'Actor_02', ...]
  Videos in Actor_01: 60

✅ Data validation PASSED!
✅ Ready to create dataloaders
```

**Nếu lỗi ở đây:**
- Kiểm tra lại `RAVDESS_PATH` ở Cell 3
- Đảm bảo có folders `Actor_01` đến `Actor_24`
- Đảm bảo mỗi folder có file `.mp4`

#### Cell 8: Create Model ✅
```python
# Tạo model
Creating model...
======================================================================
Multimodal FER Model Summary
======================================================================
Total: 149,021,194 params (149.02M)
```

#### Cell 9: Create Dataloaders ✅ KEY CELL!
```python
# Tạo dataloaders
Creating dataloaders...
Loaded 2008 videos for train split (speech)
Loaded 480 videos for val split (speech)
Loaded 480 videos for test split (speech)

✅ Dataloaders created successfully!
  Train: 2008 samples (125 batches)
  Val:   480 samples (30 batches)
  Test:  480 samples (30 batches)
```

**Nếu vẫn thấy "Loaded 0 videos":**
1. Quay lại Cell 7, check output
2. Kiểm tra `RAVDESS_PATH` ở Cell 3
3. Chạy lại Cell 3 → Cell 7 → Cell 9

#### Cell 10-12: Training Setup & Functions ✅
```python
# Setup optimizer, loss, metrics
✓ Training setup complete!
✓ Training functions defined
```

#### Cell 13: Main Training Loop 🚀
```python
# BẮT ĐẦU TRAINING - 2-3 GIỜ!
STARTING TRAINING
======================================================================
Start time: 2026-01-05 10:00:00
Total epochs: 40
======================================================================

Epoch 1/40
Training: 100%|██████████| 125/125 [03:24<00:00]
Validation: 100%|██████████| 30/30 [00:32<00:00]

Results:
  Train Loss: 1.8234 | Train Acc: 32.50%
  Val Loss: 1.6543 | Val Acc: 38.20%
  Val F1: 0.3456
  ✓ New best model! Saved to: checkpoints/ravdess_speech_t4/best_model.pt

...

Epoch 40/40
Results:
  Train Loss: 0.2134 | Train Acc: 92.50%
  Val Loss: 0.6234 | Val Acc: 78.50%
  Val F1: 0.7623
  ✓ Checkpoint saved: checkpoint_epoch_40.pt

======================================================================
TRAINING COMPLETE!
======================================================================
Duration: 2:34:15
Best Val Accuracy: 78.50%
```

#### Cell 14: Plot Results 📈
```python
# Vẽ biểu đồ
✓ Plot saved to: checkpoints/ravdess_speech_t4/training_curves.png
```

#### Cell 15: Test Evaluation 🧪
```python
# Đánh giá trên test set
EVALUATING ON TEST SET
======================================================================
✓ Loaded best model from epoch 35

Test Results:
  Loss: 0.6543
  Accuracy: 76.25%
  F1 Score: 0.7412
  Precision: 0.7523
  Recall: 0.7301
```

#### Cell 16: Download Checkpoints 💾
```python
# Download về máy
Preparing files for download...
Downloading checkpoints.zip...
✓ Download complete!
```

## 📊 Kết Quả Mong Đợi

### T4 GPU (Free Colab)
- **Thời gian**: 2-3 giờ (40 epochs)
- **Memory**: 8-10 GB VRAM
- **Accuracy**: 75-80%
- **F1 Score**: 0.73-0.78

### A100 GPU (Colab Pro)
- **Thời gian**: 1 giờ (40 epochs)
- **Memory**: 15-20 GB VRAM
- **Accuracy**: 80-85%
- **F1 Score**: 0.78-0.83

## 🔧 Troubleshooting

### Lỗi 1: "Loaded 0 videos"

**Nguyên nhân**: Đường dẫn RAVDESS sai hoặc cấu trúc folder không đúng

**Giải pháp**:
1. Kiểm tra Cell 3, sửa `RAVDESS_PATH`
2. Chạy cell này để kiểm tra:
```python
!ls -la /content/drive/MyDrive/[HUST]_Facial_Expression_Recognition/Dataset/Multimodal_DFER/RAVDESS/ | head -20
```
3. Phải thấy folders: `Actor_01`, `Actor_02`, ..., `Actor_24`

### Lỗi 2: Out of Memory (OOM)

**Nguyên nhân**: Batch size quá lớn

**Giải pháp**: Sửa Cell 6:
```python
CONFIG = {
    "batch_size": 8,  # Giảm từ 16 xuống 8
    "gradient_accumulation_steps": 2,  # Bù lại bằng accumulation
}
```

### Lỗi 3: Training quá chậm

**Giải pháp 1**: Tắt audio
```python
CONFIG = {
    "use_audio": False,  # Visual-only, nhanh hơn
}
```

**Giải pháp 2**: Giảm model size
```python
CONFIG = {
    "num_audio_layers": 6,  # Từ 8 xuống 6
    "num_visual_layers": 3,  # Từ 4 xuống 3
    "num_fusion_layers": 3,  # Từ 4 xuống 3
}
```

### Lỗi 4: Colab disconnect

**Nguyên nhân**: Session timeout (12 giờ)

**Giải pháp**:
1. Dùng Colab Pro (24 giờ)
2. Hoặc giảm epochs xuống 20
3. Checkpoints đã save, có thể resume sau

### Lỗi 5: Google Drive không mount được

**Giải pháp**:
1. Chạy lại Cell 3
2. Click vào link authorize
3. Copy code và paste vào
4. Nếu vẫn lỗi, restart runtime và chạy lại từ đầu

## 📁 Files Được Tạo Ra

Sau khi training xong, trong folder `checkpoints/ravdess_speech_t4/`:

```
checkpoints/ravdess_speech_t4/
├── best_model.pt              # Model tốt nhất (val acc cao nhất)
├── final_model.pt             # Model epoch cuối
├── checkpoint_epoch_10.pt     # Checkpoint epoch 10
├── checkpoint_epoch_20.pt     # Checkpoint epoch 20
├── checkpoint_epoch_30.pt     # Checkpoint epoch 30
├── checkpoint_epoch_40.pt     # Checkpoint epoch 40
├── config.json                # Cấu hình training
├── training_history.json      # Lịch sử metrics
├── training_curves.png        # Biểu đồ
└── test_results.json          # Kết quả test set
```

## 💡 Tips

1. **Kiểm tra GPU trước khi train**:
```python
!nvidia-smi
```

2. **Monitor VRAM usage**:
```python
!watch -n 1 nvidia-smi  # Ctrl+C để thoát
```

3. **Test với 5 epochs trước**:
```python
CONFIG["num_epochs"] = 5  # Test nhanh
```

4. **Keep Drive mounted**:
- Không unmount Drive trong khi training
- Không đóng tab Colab

5. **Save checkpoints thường xuyên**:
```python
CONFIG["save_every"] = 5  # Save mỗi 5 epochs
```

## ✅ Checklist Trước Khi Train

- [ ] Đã push code lên GitHub
- [ ] Đã mở notebook trong Colab
- [ ] Đã chọn T4 GPU runtime
- [ ] Đã mount Google Drive
- [ ] Đã cập nhật `RAVDESS_PATH` đúng
- [ ] Cell 7 validation PASSED
- [ ] Cell 9 dataloaders created (2008/480/480 samples)
- [ ] Đã đọc hướng dẫn troubleshooting
- [ ] Sẵn sàng chờ 2-3 giờ

## 🎯 Tóm Tắt

1. **Push code**: `git push origin main`
2. **Mở Colab**: https://colab.research.google.com/
3. **Chọn GPU**: T4 hoặc A100
4. **Sửa path**: Cell 3 - `RAVDESS_PATH`
5. **Chạy tất cả cells**: Shift+Enter từng cell
6. **Chờ training**: 2-3 giờ
7. **Download model**: Cell 16
8. **Xong!** 🎉

## 📞 Nếu Vẫn Gặp Lỗi

Kiểm tra lại:
1. ✅ Đã push code mới nhất lên GitHub?
2. ✅ Cell 7 validation có PASSED không?
3. ✅ Cell 9 có load được 2008 samples không?
4. ✅ `RAVDESS_PATH` có đúng không?

Nếu vẫn lỗi, chụp màn hình error và kiểm tra lại từng bước!

---

**Chúc bạn training thành công! 🚀**
