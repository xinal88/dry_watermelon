# ⚡ Quick Start: Training trên Colab IDE

## 🎯 3 Bước Đơn Giản

### 1️⃣ Upload RAVDESS lên Google Drive

```
My Drive/RAVDESS/
├── Actor_01/
├── Actor_02/
└── ... (24 actors total)
```

### 2️⃣ Mở Notebook trong IDE

- Mở file: `Train_Multimodal_FER.ipynb`
- Connect to Google Colab kernel
- Chạy tất cả cells (Run All)

### 3️⃣ Đợi Training Hoàn Thành

- ⏱️ Thời gian: 2-4 giờ
- 📊 Theo dõi UAR metric
- 💾 Tự động download checkpoints

---

## ⚙️ Cấu Hình Nhanh (Cell 6)

```python
CONFIG = {
    "batch_size": 16,          # Giảm xuống 8 nếu OOM
    "num_epochs": 100,
    "use_pretrained_visual": False,  # False = nhanh hơn
    "use_pretrained_fusion": False,  # False = nhanh hơn
}
```

---

## 📊 Kết Quả Mong Đợi

- **UAR**: 75-80% (lightweight) hoặc 80-85% (pretrained)
- **Training time**: 2-3 giờ (T4) hoặc 4-6 giờ (A100)
- **Model size**: ~150M parameters

---

## 🧪 Test Model

Sau khi training xong:

```bash
python scripts/inference_cpu.py
```

---

## ⚠️ Nếu Gặp Lỗi

- **OOM**: Giảm `batch_size` từ 16 → 8
- **RAVDESS not found**: Kiểm tra đường dẫn Drive
- **Disconnect**: Training tự động save checkpoint mỗi 10 epochs

---

**Đọc chi tiết**: `COLAB_IDE_SETUP.md`
