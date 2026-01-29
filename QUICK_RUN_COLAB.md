# ⚡ QUICK RUN - COLAB PRO (5 phút setup)

## 🎯 TÓM TẮT NHANH

Copy-paste các cells này vào Colab theo thứ tự:

---

## 📱 CELL 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🔍 CELL 2: Verify Dataset
```python
import os
from pathlib import Path

# 🔥 UPDATE YOUR PATH HERE!
RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"

if os.path.exists(RAVDESS_PATH):
    videos = list(Path(RAVDESS_PATH).rglob("*.mp4"))
    print(f"✅ Found {len(videos)} videos")
else:
    print(f"❌ Path not found: {RAVDESS_PATH}")
    print("\nAvailable folders:")
    print(os.listdir("/content/drive/MyDrive")[:10])
```

---

## 💻 CELL 3: Clone Repo
```python
!git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
%cd multimodal-fer
!ls
```

---

## 📦 CELL 4: Install Dependencies
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers timm einops opencv-python librosa soundfile tqdm scikit-learn
!apt-get install -y ffmpeg > /dev/null 2>&1

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## ⚙️ CELL 5: Update Config
```python
# Edit config file
import fileinput

# Read
with open('colab_train_easy.py', 'r') as f:
    content = f.read()

# 🔥 UPDATE YOUR PATH HERE!
YOUR_RAVDESS_PATH = "/content/drive/MyDrive/RAVDESS"

# Replace
content = content.replace(
    '"RAVDESS_PATH": "/content/drive/MyDrive/RAVDESS"',
    f'"RAVDESS_PATH": "{YOUR_RAVDESS_PATH}"'
)

# Write
with open('colab_train_easy.py', 'w') as f:
    f.write(content)

print("✅ Config updated!")

# Verify
!grep "RAVDESS_PATH" colab_train_easy.py
```

---

## 🧪 CELL 6: Quick Test (Optional, 10 phút)
```python
# Test with 5 epochs first
with open('colab_train_easy.py', 'r') as f:
    content = f.read()

content = content.replace(
    '"max_epochs": 50,',
    '"max_epochs": 5,'
)

with open('colab_train_easy.py', 'w') as f:
    f.write(content)

!python colab_train_easy.py
```

---

## 🚀 CELL 7: Full Training (1.5-2 giờ)
```python
# Change back to 50 epochs
with open('colab_train_easy.py', 'r') as f:
    content = f.read()

content = content.replace(
    '"max_epochs": 5,',
    '"max_epochs": 50,'
)

with open('colab_train_easy.py', 'w') as f:
    f.write(content)

# Start training!
!python colab_train_easy.py
```

---

## 📊 CELL 8: Monitor Progress (Trong khi training)
```python
import json
import os

history_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/training_history.json"

if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
    
    if len(history) > 0:
        last = history[-1]
        best = max(history, key=lambda x: x['val_acc'])
        
        print(f"Completed: {len(history)} epochs")
        print(f"\nLast epoch ({last['epoch']}):")
        print(f"  Train: {last['train_acc']:.2f}%")
        print(f"  Val: {last['val_acc']:.2f}%")
        print(f"\nBest epoch ({best['epoch']}):")
        print(f"  Val: {best['val_acc']:.2f}%")
else:
    print("Training not started yet or history file not created")
```

---

## 📈 CELL 9: Plot Results (Sau khi training xong)
```python
import json
import matplotlib.pyplot as plt

with open("/content/drive/MyDrive/checkpoints/multimodal_fer/training_history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_acc = [h["train_acc"] for h in history]
val_acc = [h["val_acc"] for h in history]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label="Train", marker='o')
plt.plot(epochs, val_acc, label="Val", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()

best = max(history, key=lambda x: x['val_acc'])
print(f"\n🎉 Best: Epoch {best['epoch']}, Val Acc: {best['val_acc']:.2f}%")
```

---

## 💾 CELL 10: Download Model (Optional)
```python
from google.colab import files

checkpoint_path = "/content/drive/MyDrive/checkpoints/multimodal_fer/best_model.pth"
files.download(checkpoint_path)

print("✅ Downloaded!")
```

---

## ✅ EXPECTED RESULTS

```
Training Time: ~1.5-2 hours
Final Accuracy: ~80-82% (lightweight) or ~82-85% (full)
Checkpoints: Saved to Google Drive
```

---

## 🔧 QUICK FIXES

### Out of Memory?
```python
# Edit config
"batch_size": 4,
"grad_accum_steps": 4,
```

### Dataset not found?
```python
# Check path
!ls /content/drive/MyDrive/
```

### Training too slow?
```python
# Use lightweight model
"model_type": "lightweight",
```

---

## 🎉 DONE!

Chỉ cần copy-paste 7 cells đầu tiên và chờ kết quả!

**Expected: ~80-82% accuracy trong ~2 giờ** 🚀
