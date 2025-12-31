# ✅ Training Checklist

## 📋 Trước Khi Training

### Setup Google Drive
- [ ] Tạo folder `RAVDESS` trong `My Drive`
- [ ] Upload 24 folders `Actor_01` đến `Actor_24`
- [ ] Verify: Mỗi actor có ~60 videos
- [ ] Tổng: ~1440 videos

### Setup IDE
- [ ] Cài Colab extension trong IDE
- [ ] Login Google account
- [ ] Verify có quyền truy cập Colab Pro (T4 GPU)

### Đọc Documentation
- [ ] Đọc `START_HERE.md`
- [ ] Đọc `READY_TO_TRAIN_COLAB.md`
- [ ] Hiểu cấu hình trong Cell 6

---

## 🚀 Trong Quá Trình Training

### Cell 1: Check GPU
- [ ] Chạy cell
- [ ] Verify có T4 GPU (hoặc A100)
- [ ] VRAM: ~15GB (T4) hoặc ~40GB (A100)

### Cell 2: Clone Repo (Optional)
- [ ] Skip nếu đang chạy local
- [ ] Hoặc clone từ GitHub nếu đã push

### Cell 3: Mount Drive
- [ ] Chạy cell
- [ ] Cho phép truy cập Drive
- [ ] Verify: `!ls /content/drive/MyDrive/RAVDESS`
- [ ] Thấy 24 folders Actor_XX

### Cell 4: Install Dependencies
- [ ] Chạy cell
- [ ] Đợi ~2-3 phút
- [ ] Verify: Không có error

### Cell 5: Import Libraries
- [ ] Chạy cell
- [ ] Verify: "✓ All imports successful!"

### Cell 6: Configuration ⚙️
- [ ] **EDIT CONFIG NẾU CẦN**
- [ ] Chọn batch_size (16 cho T4, 32 cho A100)
- [ ] Chọn pretrained (False = nhanh, True = tốt)
- [ ] Chọn save_dir
- [ ] Chạy cell

### Cell 7: Create Model
- [ ] Chạy cell
- [ ] Verify: Model summary hiển thị
- [ ] Check: ~150M params (lightweight) hoặc ~393M (full)

### Cell 8: Create Dataloaders
- [ ] Chạy cell
- [ ] Verify: Train ~960, Val ~240, Test ~240 samples
- [ ] Check: Không có error loading data

### Cell 9: Training Setup
- [ ] Chạy cell
- [ ] Verify: "✓ Training setup complete!"

### Cell 10: Training Functions
- [ ] Chạy cell
- [ ] Verify: "✓ Training functions defined!"

### Cell 11: Main Training Loop 🚀
- [ ] **CELL CHÍNH - MẤT 2-4 GIỜ**
- [ ] Chạy cell
- [ ] Theo dõi metrics:
  - [ ] Train Loss giảm dần
  - [ ] Val Loss giảm dần
  - [ ] UAR tăng dần
  - [ ] Thấy "🎉 New best UAR" định kỳ
- [ ] Đợi 100 epochs hoàn thành
- [ ] Verify: "TRAINING COMPLETED!"

### Cell 12: Plot Curves
- [ ] Chạy cell
- [ ] Xem đồ thị training curves
- [ ] Verify: Loss giảm, UAR tăng

### Cell 13: Evaluate Test Set
- [ ] Chạy cell
- [ ] Xem test results
- [ ] Verify: UAR >75% (lightweight) hoặc >80% (full)

### Cell 14: Download Checkpoints
- [ ] Chạy cell
- [ ] Download 4 files:
  - [ ] `best_model.pth`
  - [ ] `training_history.json`
  - [ ] `test_results.json`
  - [ ] `training_curves.png`

---

## 💾 Sau Training

### Organize Checkpoints
- [ ] Tạo folder `checkpoints/ravdess_speech_t4/`
- [ ] Copy `best_model.pth` vào folder
- [ ] Copy các files khác vào folder

### Test Model
- [ ] Mở `scripts/inference_cpu.py`
- [ ] Update `CHECKPOINT_PATH`
- [ ] Chạy: `python scripts/inference_cpu.py`
- [ ] Verify: Model load thành công
- [ ] Test với video mẫu

### Review Results
- [ ] Mở `training_history.json`
- [ ] Check best UAR epoch
- [ ] Mở `test_results.json`
- [ ] Verify test metrics
- [ ] Xem `training_curves.png`

---

## 📊 Metrics Checklist

### Training Progress
- [ ] Epoch 1: UAR ~30-40%
- [ ] Epoch 10: UAR ~60-65%
- [ ] Epoch 50: UAR ~70-75%
- [ ] Epoch 100: UAR ~75-80% (lightweight) hoặc ~80-85% (full)

### Final Results
- [ ] Test UAR: >75% (lightweight) hoặc >80% (full)
- [ ] Test Accuracy: >78% (lightweight) hoặc >83% (full)
- [ ] Per-class metrics: Balanced (không có class nào quá thấp)

---

## ⚠️ Troubleshooting Checklist

### Nếu OOM (Out of Memory)
- [ ] Giảm `batch_size` từ 16 → 8
- [ ] Tăng `gradient_accumulation_steps` từ 1 → 2
- [ ] Giảm `num_audio_layers` từ 8 → 6
- [ ] Restart runtime và chạy lại

### Nếu RAVDESS Not Found
- [ ] Check: `!ls /content/drive/MyDrive/RAVDESS`
- [ ] Verify: 24 folders Actor_XX
- [ ] Fix symlink: `!ln -sf /content/drive/MyDrive/RAVDESS data/ravdess`
- [ ] Chạy lại Cell 8

### Nếu Colab Disconnect
- [ ] Training đã save checkpoint mỗi 10 epochs
- [ ] Reconnect to runtime
- [ ] Chạy lại từ Cell 11
- [ ] Model sẽ resume từ checkpoint cuối

### Nếu Training Quá Chậm
- [ ] Check GPU: `!nvidia-smi`
- [ ] Verify: GPU utilization >80%
- [ ] Giảm `num_workers` từ 2 → 0
- [ ] Enable `use_amp`: True

---

## 🎯 Success Criteria

### Training Thành Công Khi:
- [x] Tất cả 14 cells chạy không lỗi
- [x] Training hoàn thành 100 epochs
- [x] Best UAR >75% (lightweight) hoặc >80% (full)
- [x] Test UAR tương đương Val UAR (±2%)
- [x] Checkpoints download thành công
- [x] Inference chạy được trên local

### Có Thể Cải Thiện Nếu:
- [ ] UAR <75%: Tăng epochs, tune hyperparameters
- [ ] Overfitting: Thêm dropout, data augmentation
- [ ] Underfitting: Tăng model size, pretrained models
- [ ] Imbalanced: Adjust class weights

---

## 📝 Notes

### Training Time
- T4 GPU: 2-3 giờ (lightweight)
- A100 GPU: 4-6 giờ (full pretrained)

### Model Size
- Lightweight: ~150M params, ~600MB file
- Full: ~393M params, ~1.5GB file

### Expected UAR
- Lightweight: 75-80%
- Full Pretrained: 80-85%
- State-of-the-art: 85-90%

---

## ✅ Final Checklist

- [ ] Training completed successfully
- [ ] UAR >75% achieved
- [ ] Checkpoints downloaded
- [ ] Inference tested on local
- [ ] Results documented
- [ ] Ready for deployment

---

**Hoàn thành tất cả checklist = Training thành công! 🎉**
