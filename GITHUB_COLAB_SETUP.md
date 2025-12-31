# 🚀 Setup GitHub + Colab - Hướng Dẫn Chi Tiết

## 📋 Tổng Quan

Sử dụng GitHub để quản lý code và Google Colab để train model.

---

## 🎯 Bước 1: Chuẩn Bị GitHub Repository

### 1.1. Tạo Repository Mới

```bash
# Trên máy local (trong thư mục dry_watermelon)
git init
git add .
git commit -m "Initial commit: Multimodal FER project"

# Tạo repo trên GitHub: https://github.com/new
# Tên repo: multimodal-fer (hoặc tên bạn thích)

# Link local với GitHub
git remote add origin https://github.com/YOUR_USERNAME/multimodal-fer.git
git branch -M main
git push -u origin main
```

### 1.2. Tạo .gitignore

Tạo file `.gitignore` để không push file không cần thiết:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data (quá lớn, để trên Drive)
data/ravdess/
data/Video_Song_Actor_*/
*.mp4
*.wav

# Checkpoints (sẽ download từ Colab)
checkpoints/
*.pth
*.pt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
temp/
```

### 1.3. Push lên GitHub

```bash
git add .gitignore
git commit -m "Add .gitignore"
git push
```

---

## 💾 Bước 2: Chuẩn Bị Dữ Liệu trên Google Drive

### 2.1. Cấu Trúc Thư Mục

```
My Drive/
└── RAVDESS/
    ├── Video_Speech_Actor_01/
    ├── Video_Speech_Actor_02/
    ├── ...
    └── Video_Speech_Actor_24/
```

### 2.2. Upload Dữ Liệu

- Upload toàn bộ RAVDESS dataset lên Drive
- Đường dẫn: `My Drive/RAVDESS/`
- Khoảng 1440 videos (~10-15GB)

---

## 📓 Bước 3: Tạo Colab Notebook

### 3.1. Mở Google Colab

1. Truy cập: https://colab.research.google.com
2. File → New Notebook
3. Đổi tên: `Train_Multimodal_FER.ipynb`

### 3.2. Chọn GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (Free) hoặc **A100** (Pro)
4. Save

### 3.3. Copy Code vào Notebook

**Option A: Copy từ file Python**

Mở file `colab_train.py` và copy từng cell (phần giữa `# %%`) vào Colab.

**Option B: Tạo từ template**

Tôi sẽ tạo file notebook template cho bạn (xem bên dưới).

---

## 🔧 Bước 4: Chạy Training trên Colab

### 4.1. Cell 1: Check GPU

```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4.2. Cell 2: Clone Repository

```python
# Clone từ GitHub
!git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
%cd multimodal-fer

# Hoặc nếu đã clone, pull latest
# !git pull origin main
```

### 4.3. Cell 3: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Link RAVDESS data
!ln -s /content/drive/MyDrive/RAVDESS data/ravdess
!ls data/ravdess | head -10
```

### 4.4. Cell 4: Install Dependencies

```python
!pip install -q transformers==4.36.0 einops scikit-learn matplotlib seaborn
!which ffmpeg  # Should be available
```

### 4.5. Cell 5-14: Copy từ colab_train.py

Copy các cells còn lại từ file `colab_train.py`.

---

## 📊 Bước 5: Monitor Training

### 5.1. Trong Colab

- Xem progress bars (tqdm)
- Xem metrics sau mỗi epoch
- Xem training curves

### 5.2. TensorBoard (Optional)

```python
# Thêm vào training loop
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/{CONFIG["save_dir"]}')

# Log metrics
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Metrics/UAR', val_metrics['uar'], epoch)
```

```python
# Xem trong Colab
%load_ext tensorboard
%tensorboard --logdir runs
```

---

## 💾 Bước 6: Download Checkpoints

### 6.1. Download trực tiếp từ Colab

```python
from google.colab import files

# Download best model
files.download("checkpoints/ravdess_speech_t4/best_model.pth")

# Download training history
files.download("checkpoints/ravdess_speech_t4/training_history.json")
```

### 6.2. Hoặc Copy sang Drive

```python
# Copy checkpoints sang Drive
!cp -r checkpoints /content/drive/MyDrive/RAVDESS_Checkpoints/
```

---

## 🔄 Bước 7: Sync về Local Machine

### 7.1. Download từ Drive

- Mở Google Drive
- Tìm folder `RAVDESS_Checkpoints`
- Download `best_model.pth`

### 7.2. Hoặc Clone từ GitHub (nếu đã push)

```bash
# Trên local machine
cd dry_watermelon
git pull origin main

# Checkpoints sẽ ở checkpoints/
```

---

## 🧪 Bước 8: Test trên Local

### 8.1. Copy Checkpoint

```bash
# Copy best_model.pth vào local
cp ~/Downloads/best_model.pth checkpoints/ravdess_speech_t4/
```

### 8.2. Run Inference

```bash
# Edit CONFIG trong scripts/inference_cpu.py
python scripts/inference_cpu.py
```

### 8.3. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ravdess_speech_t4/best_model.pth \
    --data-dir data/ravdess \
    --split test
```

---

## 📝 Workflow Hoàn Chỉnh

```
┌─────────────────────────────────────────────────────────┐
│ 1. LOCAL: Develop & Test                               │
│    - Write code                                         │
│    - Test with 3 videos                                 │
│    - Push to GitHub                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. GITHUB: Version Control                             │
│    - Store code                                         │
│    - Track changes                                      │
│    - Collaborate                                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. COLAB: Train with GPU                               │
│    - Clone from GitHub                                  │
│    - Mount Drive (data)                                 │
│    - Train 2-4 hours                                    │
│    - Download checkpoints                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. LOCAL: Inference & Deploy                           │
│    - Load trained model                                 │
│    - Test inference                                     │
│    - Deploy to production                               │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 Tips & Best Practices

### 1. Quản Lý Code

```bash
# Tạo branch mới cho experiments
git checkout -b experiment/new-architecture

# Commit thường xuyên
git add .
git commit -m "Add: new feature"
git push origin experiment/new-architecture

# Merge khi thành công
git checkout main
git merge experiment/new-architecture
```

### 2. Quản Lý Checkpoints

```python
# Đặt tên checkpoint có ý nghĩa
save_dir = f"checkpoints/ravdess_{modality}_{gpu_type}_{timestamp}"

# Lưu config cùng checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": CONFIG,  # Quan trọng!
    "metrics": metrics,
}
```

### 3. Tối Ưu Training

```python
# Sử dụng gradient accumulation nếu OOM
CONFIG["batch_size"] = 8
CONFIG["gradient_accumulation_steps"] = 4
# Effective batch size = 8 * 4 = 32

# Checkpoint thường xuyên
CONFIG["save_every"] = 5  # Save every 5 epochs

# Early stopping
if epoch > 20 and val_metrics["uar"] < 0.5:
    print("Early stopping: UAR too low")
    break
```

### 4. Debug trên Colab

```python
# Test với 1 batch trước
for audio, video, labels, _ in train_loader:
    outputs = model(audio.cuda(), video.cuda())
    print(f"Output shape: {outputs['logits'].shape}")
    break

# Giảm epochs để test nhanh
CONFIG["num_epochs"] = 5  # Test run
```

---

## 🐛 Troubleshooting

### Issue 1: "Repository not found"

```bash
# Check remote URL
git remote -v

# Update URL
git remote set-url origin https://github.com/YOUR_USERNAME/multimodal-fer.git
```

### Issue 2: "CUDA out of memory"

```python
# Giảm batch size
CONFIG["batch_size"] = 4

# Hoặc dùng gradient accumulation
CONFIG["gradient_accumulation_steps"] = 4
```

### Issue 3: "Drive mount failed"

```python
# Unmount và mount lại
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

### Issue 4: "ffmpeg not found"

```python
# Colab có sẵn ffmpeg, nhưng nếu lỗi:
!apt-get install -y ffmpeg

# Hoặc tạm thời không dùng audio
CONFIG["use_audio"] = False
```

---

## ✅ Checklist

**Trước khi train:**
- [ ] Code đã push lên GitHub
- [ ] Dữ liệu RAVDESS đã upload lên Drive
- [ ] Đã tạo Colab notebook
- [ ] Đã chọn GPU runtime
- [ ] Đã test clone repository
- [ ] Đã test mount Drive

**Trong khi train:**
- [ ] Monitor training progress
- [ ] Check UAR tăng dần
- [ ] Không có OOM errors
- [ ] Checkpoints được lưu

**Sau khi train:**
- [ ] Download best_model.pth
- [ ] Download training_history.json
- [ ] Test inference trên local
- [ ] Push checkpoints lên GitHub (optional)

---

## 🎉 Hoàn Thành!

Bạn đã có workflow hoàn chỉnh:

1. ✅ GitHub để quản lý code
2. ✅ Google Drive để lưu data
3. ✅ Google Colab để train
4. ✅ Local machine để test

**Sẵn sàng train! 🚀**

---

## 📞 Quick Commands

```bash
# LOCAL: Push code
git add .
git commit -m "Update model"
git push

# COLAB: Clone và train
!git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
%cd multimodal-fer
# ... run training cells ...

# LOCAL: Pull checkpoints (if pushed)
git pull origin main
```

**Good luck with training!** 🎯
