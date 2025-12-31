# 🎮 GPU Setup Guide

## Current Status

**Your Hardware:**
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4GB
- CUDA Version: 12.7
- Driver: 566.07

**Problem:**
- PyTorch is CPU-only (`2.9.1+cpu`)
- Cannot use GPU for training

---

## 🔧 Fix: Install CUDA PyTorch

### Step 1: Uninstall CPU Version

```powershell
pip uninstall torch torchvision torchaudio
```

### Step 2: Install CUDA Version

```powershell
# For CUDA 12.x (your version: 12.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected Output:**
```
PyTorch: 2.x.x+cu121
CUDA: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## ⚠️ Important: 4GB VRAM Limitation

Your RTX 3050 Laptop has **4GB VRAM**, which is limited for large models.

### **Model Size Estimates:**

| Configuration | VRAM Usage | Fits 4GB? |
|---------------|------------|-----------|
| **Full Pretrained** (SigLIP2 + LFM2-700M) | ~6-8GB | ❌ No |
| **Hybrid** (Custom CNN + LFM2-700M) | ~4-5GB | ⚠️ Tight |
| **Lightweight** (Custom CNN + Custom LFM2) | ~2-3GB | ✅ Yes |

---

## 🎯 Recommended Configuration for 4GB

### **Option 1: Lightweight (Recommended)**

Use custom implementations for both vision and fusion:

```python
# In your training script or config
from models import (
    MultimodalFER,
    VisualBranchConfig,
    LFM2FusionConfig,
)

# Lightweight config
visual_config = VisualBranchConfig(
    use_pretrained_encoder=False,  # Custom CNN
    feature_dim=768,
)

fusion_config = LFM2FusionConfig(
    use_pretrained=False,  # Custom LFM2
    num_layers=4,  # Reduce layers
    hidden_dim=1024,  # Reduce hidden size
)

model = MultimodalFER(
    visual_config=visual_config,
    fusion_config=fusion_config,
    num_classes=8,
)
```

**Benefits:**
- ✅ Fits comfortably in 4GB
- ✅ Fast training
- ✅ No downloads needed
- ✅ End-to-end trainable
- ✅ Still good performance (75-80% UAR expected)

### **Option 2: Hybrid (If you want pretrained)**

Use custom CNN but keep pretrained LFM2:

```python
visual_config = VisualBranchConfig(
    use_pretrained_encoder=False,  # Custom CNN (saves ~2GB)
)

fusion_config = LFM2FusionConfig(
    use_pretrained=True,  # Pretrained LFM2
    freeze_backbone=True,  # Freeze to save memory
    num_layers=4,  # Use fewer layers
)

model = MultimodalFER(
    visual_config=visual_config,
    fusion_config=fusion_config,
)
```

**Training Settings for 4GB:**
```python
config = {
    "batch_size": 1,  # Small batch
    "gradient_accumulation_steps": 4,  # Simulate batch_size=4
    "mixed_precision": True,  # Use FP16
}
```

---

## 🚀 Quick Start Commands

### **1. Install CUDA PyTorch:**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **2. Verify GPU:**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### **3. Create Lightweight Config:**

Create `configs/lightweight_config.py`:
```python
from models import VisualBranchConfig, LFM2FusionConfig

# Optimized for 4GB VRAM
VISUAL_CONFIG = VisualBranchConfig(
    use_pretrained_encoder=False,
    feature_dim=512,  # Reduced
    temporal_depth=4,  # Reduced
)

FUSION_CONFIG = LFM2FusionConfig(
    use_pretrained=False,
    num_layers=4,
    hidden_dim=1024,
)

TRAINING_CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": True,
}
```

### **4. Train with Lightweight Config:**
```bash
python scripts/train_test_samples.py
```

---

## 📊 Memory Optimization Tips

### **1. Reduce Batch Size**
```python
batch_size = 1  # Minimum for 4GB
```

### **2. Use Gradient Accumulation**
```python
# Simulate larger batch size
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps  # = 4
```

### **3. Enable Mixed Precision (FP16)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(audio, video)
```

### **4. Clear Cache Regularly**
```python
torch.cuda.empty_cache()
```

### **5. Reduce Model Size**
```python
# Fewer layers
temporal_depth = 4  # instead of 6
num_layers = 4  # instead of 6

# Smaller hidden dimensions
hidden_dim = 1024  # instead of 1536
feature_dim = 512  # instead of 768
```

---

## 🎯 Expected Performance

### **Lightweight Config (4GB friendly):**
- **VRAM Usage**: ~2-3GB
- **Training Speed**: ~2-3 min/epoch (GPU)
- **Expected UAR**: 75-80%
- **Batch Size**: 1-2

### **Full Config (Needs 8GB+):**
- **VRAM Usage**: ~6-8GB
- **Training Speed**: ~1-2 min/epoch (GPU)
- **Expected UAR**: 80-85%
- **Batch Size**: 4-8

---

## 🔍 Troubleshooting

### **"CUDA out of memory" Error:**

**Solution 1: Reduce batch size**
```python
batch_size = 1
```

**Solution 2: Use custom models**
```python
use_pretrained_encoder = False
use_pretrained = False
```

**Solution 3: Reduce model size**
```python
temporal_depth = 2
num_layers = 2
hidden_dim = 512
```

### **"CUDA available: False" after installation:**

**Check PyTorch version:**
```powershell
python -c "import torch; print(torch.__version__)"
```

Should show `+cu121`, not `+cpu`.

**If still CPU version:**
```powershell
# Completely remove and reinstall
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ✅ Summary

**Your Situation:**
- ✅ GPU available: RTX 3050 (4GB)
- ✅ CUDA installed: 12.7
- ❌ PyTorch: CPU-only version

**Action Required:**
1. Install CUDA PyTorch
2. Use lightweight config for 4GB
3. Train with batch_size=1

**Commands:**
```powershell
# Install CUDA PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Train (will auto-detect GPU)
python scripts/train_test_samples.py
```

---

## 🎉 After Setup

Once CUDA PyTorch is installed:
- ✅ GPU will be automatically detected
- ✅ Training will be 10-20x faster
- ✅ Model will use GPU memory
- ✅ Mixed precision will work

**Training time comparison:**
- CPU: ~30-60 minutes (50 epochs)
- GPU: ~5-10 minutes (50 epochs)

---

**Ready to install? Run the commands above!** 🚀
