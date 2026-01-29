# 🚀 HƯỚNG DẪN CHẠY TRÊN COLAB PRO - TỪNG BƯỚC

## ✅ CHECKLIST TRƯỚC KHI BẮT ĐẦU

- [ ] Có tài khoản Google Colab Pro
- [ ] Đã upload RAVDESS dataset lên Google Drive (~3GB)
- [ ] Đã có GitHub repository với code này

---

## 📋 BƯỚC 1: TẠO COLAB NOTEBOOK (2 phút)

### 1.1. Mở Google Colab
1. Truy cập: https://colab.research.google.com/
2. Đăng nhập với tài khoản Google của bạn

### 1.2. Tạo Notebook Mới
1. Click **File > New notebook**
2. Đổi tên notebook: **Multimodal_FER_Training**
3. Click vào tên để đổi

### 1.3. Chọn GPU Runtime
1. Click **Runtime > Change runtime type**
2. Chọn:
   - **Hardware accelerator**: GPU
   - **GPU type**: A100 (nếu có) hoặc V100
3. Click **Save**

### 1.4. Verify GPU
Tạo cell mới và chạy:
```python
!nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    44W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ Nếu thấy GPU name (A100, V100, T4), bạn đã sẵn sàng!

---

## 📦 BƯỚC 2: MOUNT GOOGLE DRIVE (1 phút)

### 2.1. Mount Drive
Tạo cell mới:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2. Authorize
1. Click vào link xuất hiện
2. Chọn tài khoản Google
3. Click **Allow**
4. Copy authorization code
5. Paste vào ô input trong Colab
6. Press Enter

**Expected output:**
```
Mounted at /content/drive
```

### 2.3. Verify Drive
```python
import os
print("Drive contents:", os.listdir('/content/drive/MyDrive')[:10])
```

✅ Nếu thấy danh sách folders, Drive đã mount thành công!

---

## 🗂️ BƯỚC 3: VERIFY RAVDESS DATASET (2 phút)

### 3.1. Check Dataset Path
```python
import os
from pathlib import Path

# 🔥 UPDATE THIS PATH!
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"

# Check if exists
if os.path.exists(RAVDESS_PATH):
    print(f"✅ RAVDESS found at: {RAVDESS_PATH}")
    
    # Count videos
    videos = list(Path(RAVDESS_PATH).rglob("*.mp4"))
    print(f"✅ Total videos: {len(videos)}")
    
    # Show first 5
    print(f"\nFirst 5 videos:")
    for v in videos[:5]:
        print(f"  - {v.name}")
else:
    print(f"❌ RAVDESS NOT FOUND at: {RAVDESS_PATH}")
    print(f"\nAvailable folders in MyDrive:")
    for item in os.listdir("/content/drive/MyDrive")[:20]:
        print(f"  - {item}")
```

**Expected output:**
```
✅ RAVDESS found at: /content/drive/MyDrive/RAVDESS
✅ Total videos: 1440

First 5 videos:
  - 01-01-01-01-01-01-01.mp4
  - 01-01-01-01-01-02-01.mp4
  - 01-01-01-01-02-01-01.mp4
  - 01-01-01-01-02-02-01.mp4
  - 01-01-01-02-01-01-01.mp4
```

### 3.2. Nếu Path Sai
Nếu không tìm thấy, update `RAVDESS_PATH`:
```python
# Ví dụ các path phổ biến:
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"
RAVDESS_PATH = "/content/drive/MyDrive/Datasets/RAVDESS"
RAVDESS_PATH = "/content/drive/MyDrive/Data/RAVDESS"
```

✅ Khi thấy "Total videos: 1440", bạn đã sẵn sàng!

---

## 💻 BƯỚC 4: CLONE REPOSITORY (2 phút)

### 4.1. Clone Repo
```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
%cd multimodal-fer

print("✅ Repository cloned!")
```

**Thay `YOUR_USERNAME` bằng username GitHub của bạn!**

### 4.2. Verify Structure
```python
!ls -la
```

**Expected output:**
```
total 156
drwxr-xr-x  8 root root  4096 Jan 29 10:30 .
drwxr-xr-x  1 root root  4096 Jan 29 10:30 ..
drwxr-xr-x  8 root root  4096 Jan 29 10:30 .git
-rw-r--r--  1 root root   123 Jan 29 10:30 .gitignore
-rw-r--r--  1 root root  5432 Jan 29 10:30 colab_train_easy.py
drwxr-xr-x  3 root root  4096 Jan 29 10:30 data
drwxr-xr-x  5 root root  4096 Jan 29 10:30 models
drwxr-xr-x  2 root root  4096 Jan 29 10:30 scripts
...
```

✅ Nếu thấy `colab_train_easy.py`, `data/`, `models/`, bạn đã sẵn sàng!

---

## 📚 BƯỚC 5: INSTALL DEPENDENCIES (3 phút)

### 5.1. Install PyTorch
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5.2. Install Other Dependencies
```python
!pip install -q transformers timm einops
!pip install -q opencv-python librosa soundfile
!pip install -q tqdm scikit-learn
```

### 5.3. Install FFmpeg (for audio extraction)
```python
!apt-get install -y ffmpeg > /dev/null 2>&1
print("✅ FFmpeg installed!")
```

### 5.4. Verify Installation
```python
import torch
import transformers
import cv2
import librosa

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ Librosa: {librosa.__version__}")
```

**Expected output:**
```
✅ PyTorch: 2.1.0+cu118
✅ CUDA available: True
✅ Transformers: 4.36.0
✅ OpenCV: 4.8.0
✅ Librosa: 0.10.1
```

✅ Tất cả dependencies đã sẵn sàng!

---

## ⚙️ BƯỚC 6: CONFIGURE TRAINING SCRIPT (1 phút)

### 6.1. Open Configuration File
```python
# View current config
!head -60 colab_train_easy.py | tail -30
```

### 6.2. Update RAVDESS Path
```python
# Edit the config
import fileinput

# Read file
with open('colab_train_easy.py', 'r') as f:
    content = f.read()

# Replace path (UPDATE THIS!)
OLD_PATH = '"/content/drive/MyDrive/RAVDESS"'
NEW_PATH = '"/content/drive/MyDrive/RAVDESS"'  # Your actual path

content = content.replace(OLD_PATH, NEW_PATH)

# Write back
with open('colab_train_easy.py', 'w') as f:
    f.write(content)

print("✅ Configuration updated!")
```

**HOẶC** edit trực tiếp trong Colab:
1. Click vào folder icon bên trái
2. Navigate to `colab_train_easy.py`
3. Double click để mở
4. Tìm dòng `"RAVDESS_PATH": "/content/drive/MyDrive/RAVDESS"`
5. Thay đổi path
6. Ctrl+S để save

### 6.3. Verify Config
```python
# Check config
!grep "RAVDESS_PATH" colab_train_easy.py
```

**Expected output:**
```
    "RAVDESS_PATH": "/content/drive/MyDrive/RAVDESS",
```

✅ Configuration đã sẵn sàng!

---

## 🧪 BƯỚC 7: TEST DATASET LOADER (2 phút)

### 7.1. Quick Test
```python
# Test dataset loader
from data.simple_ravdess_dataset import SimpleRAVDESSDataset

# Update path
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"

# Create dataset
dataset = SimpleRAVDESSDataset(
    data_dir=RAVDESS_PATH,
    split="train",
    modality="speech",
    use_audio=True,
)

print(f"✅ Dataset loaded: {len(dataset)} samples")

# Test loading one sample
audio, video, label, metadata = dataset[0]

print(f"\nSample 0:")
print(f"  Audio shape: {audio.shape}")
print(f"  Video shape: {video.shape}")
print(f"  Label: {label} ({dataset.emotion_names[label]})")
print(f"  Filename: {metadata['filename']}")
```

**Expected output:**
```
Loaded 960 videos for train split (speech)
✅ Dataset loaded: 960 samples

Sample 0:
  Audio shape: torch.Size([48000])
  Video shape: torch.Size([16, 3, 224, 224])
  Label: 0 (neutral)
  Filename: 01-01-01-01-01-01-01.mp4
```

✅ Dataset loader hoạt động!

---

## 🚀 BƯỚC 8: START TRAINING! (1.5-2 giờ)

### 8.1. Quick Test (5 epochs, 10 phút)
```python
# Quick test first
!python colab_train_easy.py
```

Sau đó edit `colab_train_easy.py`:
```python
"max_epochs": 5,  # Change from 50 to 5
```

Chạy lại:
```python
!python colab_train_easy.py
```

**Expected output:**
```
================================================================================
🚀 MULTIMODAL FER TRAINING - EASY COLAB SETUP
================================================================================

✅ Device: cuda
✅ GPU: Tesla A100-SXM4-40GB
✅ VRAM: 40.0 GB

📁 Checking RAVDESS dataset...
   Path: /content/drive/MyDrive/RAVDESS
✅ Found 1440 total videos
✅ Save directory: /content/drive/MyDrive/checkpoints/multimodal_fer

================================================================================
✅ SETUP COMPLETE - READY TO TRAIN!
================================================================================

[1/4] Creating model...
...
```

### 8.2. Full Training (50 epochs)
Nếu quick test thành công, edit lại:
```python
"max_epochs": 50,  # Back to 50
```

Chạy full training:
```python
!python colab_train_easy.py
```

**Training sẽ chạy ~1.5-2 giờ**

---

## 📊 BƯỚC 9: MONITOR TRAINING (Trong khi chờ)

### 9.1. Watch Progress
Training progress sẽ hiển thị real-time:
```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|██████████| 120/120 [01:02<00:00, 1.92it/s, loss=1.8234, acc=35.23%]
Validation: 100%|██████████| 24/24 [00:08<00:00, 2.85it/s]

Results:
  Train Loss: 1.8234, Acc: 35.23%
  Val Loss: 1.6543, Acc: 42.15%
  LR: 1.00e-04
  🎉 New best! Val Acc: 42.15%
  ✅ Saved: best_model.pth
```

### 9.2. Check Checkpoints
Trong cell khác (không interrupt training):
```python
import os
checkpoint_dir = "/content/drive/MyDrive/checkpoints/multimodal_fer"

if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    print(f"Checkpoints ({len(files)} files):")
    for f in sorted(files):
        print(f"  - {f}")
```

### 9.3. View Training History
```python
import json

history_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/training_history.json"

if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
    
    print(f"Completed epochs: {len(history)}")
    
    if len(history) > 0:
        last = history[-1]
        print(f"\nLast epoch ({last['epoch']}):")
        print(f"  Train Acc: {last['train_acc']:.2f}%")
        print(f"  Val Acc: {last['val_acc']:.2f}%")
        
        best = max(history, key=lambda x: x['val_acc'])
        print(f"\nBest epoch ({best['epoch']}):")
        print(f"  Val Acc: {best['val_acc']:.2f}%")
```

---

## ✅ BƯỚC 10: VERIFY RESULTS (5 phút)

### 10.1. Check Final Output
Sau khi training xong, bạn sẽ thấy:
```
================================================================================
✅ TRAINING COMPLETED!
================================================================================

Results:
  Best Val Acc: 82.15%
  Test Acc: 80.45%
  Total Epochs: 50

Checkpoints saved in: /content/drive/MyDrive/checkpoints/multimodal_fer
================================================================================
```

### 10.2. List Checkpoints
```python
!ls -lh /content/drive/MyDrive/checkpoints/multimodal_fer/
```

**Expected:**
```
-rw------- 1 root root 486M Jan 29 12:30 best_model.pth
-rw------- 1 root root 486M Jan 29 11:45 checkpoint_epoch_5.pth
-rw------- 1 root root 486M Jan 29 11:50 checkpoint_epoch_10.pth
...
-rw------- 1 root root  15K Jan 29 12:30 training_history.json
```

### 10.3. Load Best Model
```python
import torch
from models import MultimodalFER

# Load model
model = MultimodalFER(num_classes=8, num_segments=8)

# Load checkpoint
checkpoint_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/best_model.pth"
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
print(f"✅ Val accuracy: {checkpoint['val_acc']:.2f}%")
```

---

## 📈 BƯỚC 11: VISUALIZE RESULTS (5 phút)

### 11.1. Plot Training Curves
```python
import json
import matplotlib.pyplot as plt

# Load history
with open("/content/drive/MyDrive/checkpoints/multimodal_fer/training_history.json") as f:
    history = json.load(f)

# Extract data
epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
train_acc = [h["train_acc"] for h in history]
val_acc = [h["val_acc"] for h in history]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss
ax1.plot(epochs, train_loss, label="Train Loss", marker='o')
ax1.plot(epochs, val_loss, label="Val Loss", marker='s')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(epochs, train_acc, label="Train Acc", marker='o')
ax2.plot(epochs, val_acc, label="Val Acc", marker='s')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print best
best_epoch = max(history, key=lambda x: x['val_acc'])
print(f"\n🎉 Best Epoch: {best_epoch['epoch']}")
print(f"   Val Acc: {best_epoch['val_acc']:.2f}%")
print(f"   Train Acc: {best_epoch['train_acc']:.2f}%")
```

---

## 💾 BƯỚC 12: DOWNLOAD CHECKPOINTS (Optional)

### 12.1. Download Best Model
```python
from google.colab import files

# Download best model
checkpoint_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/best_model.pth"
files.download(checkpoint_path)

print("✅ Downloaded best_model.pth")
```

### 12.2. Download Training History
```python
history_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/training_history.json"
files.download(history_path)

print("✅ Downloaded training_history.json")
```

---

## 🎉 HOÀN THÀNH!

### Tóm Tắt:
- ✅ Model trained: ~80-82% accuracy
- ✅ Checkpoints saved to Google Drive
- ✅ Training history saved
- ✅ Ready for inference!

### Next Steps:
1. **Evaluate on test set** (đã tự động chạy)
2. **Try different hyperparameters**
3. **Train with full config** (model_type="full")
4. **Deploy model for inference**

---

## 🐛 TROUBLESHOOTING

### Issue 1: "RAVDESS path not found"
```python
# Check available paths
!ls /content/drive/MyDrive/
```

### Issue 2: "Out of Memory"
Edit config:
```python
"batch_size": 4,
"grad_accum_steps": 4,
```

### Issue 3: "No videos found"
```python
# Check video files
!find /content/drive/MyDrive/RAVDESS -name "*.mp4" | head -10
```

### Issue 4: Training interrupted
Resume from checkpoint:
```python
# Load checkpoint and continue
# (Feature to be added)
```

---

## 📞 SUPPORT

Nếu gặp vấn đề:
1. Check error message
2. Verify dataset path
3. Check GPU availability
4. Review configuration

**Chúc bạn training thành công! 🚀**
