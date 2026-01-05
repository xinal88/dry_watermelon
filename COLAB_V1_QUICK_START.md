# 🚀 Quick Start - train_dry_watermelon_v1.ipynb

## ✅ Đã Fix

- ✅ Lỗi "Loaded 0 videos" 
- ✅ Mount Drive trong Colab IDE
- ✅ Validation dữ liệu trước khi train
- ✅ Error handling tốt hơn

## 🎯 5 Bước Nhanh

### 1. Push Code
```bash
git add data/ravdess_dataset.py train_dry_watermelon_v1.ipynb
git commit -m "Add v1 notebook with fixes"
git push origin main
```

### 2. Mở Colab
- Vào: https://colab.research.google.com/
- File → Open → GitHub → `xinal88/dry_watermelon`
- Chọn: `train_dry_watermelon_v1.ipynb`

### 3. Chọn GPU
- Runtime → Change runtime type → T4 GPU → Save

### 4. Sửa Path (Cell 3)
```python
# CẬP NHẬT ĐƯỜNG DẪN NÀY!
RAVDESS_PATH = "/content/drive/MyDrive/[HUST]_Facial_Expression_Recognition/Dataset/Multimodal_DFER/RAVDESS"
```

### 5. Chạy Tất Cả Cells
- Runtime → Run all
- Hoặc Shift+Enter từng cell

## ✅ Kiểm Tra Quan Trọng

### Cell 7: Validation
```
✅ Data validation PASSED!
✅ Ready to create dataloaders
```

### Cell 9: Dataloaders
```
✅ Dataloaders created successfully!
  Train: 2008 samples (125 batches)
  Val:   480 samples (30 batches)
  Test:  480 samples (30 batches)
```

**Nếu thấy 2 dòng này → OK, tiếp tục!**

## ⏱️ Thời Gian

- **T4 GPU**: 2-3 giờ (40 epochs)
- **A100 GPU**: 1 giờ (40 epochs)

## 🎯 Kết Quả Mong Đợi

- **Accuracy**: 75-80%
- **F1 Score**: 0.73-0.78

## 🔧 Nếu Lỗi

### "Loaded 0 videos"
→ Kiểm tra lại `RAVDESS_PATH` ở Cell 3

### Out of Memory
→ Cell 6: `"batch_size": 8`

### Quá chậm
→ Cell 6: `"use_audio": False`

## 📁 Files Tạo Ra

```
checkpoints/ravdess_speech_t4/
├── best_model.pt           # ← Dùng file này
├── training_curves.png
├── test_results.json
└── ...
```

## 📖 Hướng Dẫn Chi Tiết

Xem: `HUONG_DAN_COLAB_V1.md`

---

**Sẵn sàng train! 🎉**
