# 🔧 COLAB TRAINING FIXES

## ❌ ISSUES PHÁT HIỆN

### Issue 1: Audio Extraction Failing
**Triệu chứng:**
```
UserWarning: Failed to extract audio from ... : Could not load libtorchcodec
```

**Nguyên nhân:**
- torchaudio không tương thích với PyTorch 2.9.0
- FFmpeg không được cài đúng cách
- torchcodec missing

**Hậu quả:**
- Model train với silent audio (zeros)
- Accuracy thấp vì thiếu audio features

### Issue 2: NaN Loss
**Triệu chứng:**
```
loss=nan, acc=0.00%
```

**Nguyên nhân:**
- Gradient explosion
- Numerical instability
- Silent audio causing division by zero

---

## ✅ FIXES ĐÃ APPLY

### Fix 1: Robust Audio Extraction (3 fallback methods)

**File**: `data/simple_ravdess_dataset.py`

**Changes:**
```python
def _extract_audio(self, video_path: Path) -> torch.Tensor:
    # Method 1: ffmpeg + torchaudio (preferred)
    try:
        subprocess + torchaudio.load()
    except:
        # Method 2: ffmpeg + soundfile (fallback 1)
        try:
            subprocess + soundfile.read()
        except:
            # Method 3: librosa direct (fallback 2)
            try:
                librosa.load()
            except:
                # Last resort: silent audio with warning
                return torch.zeros(target_length)
```

**Benefits:**
- ✅ Multiple fallback methods
- ✅ Works even if torchaudio fails
- ✅ Clear warnings when audio extraction fails
- ✅ Graceful degradation

### Fix 2: NaN Loss Prevention

**File**: `colab_train_easy.py`

**Changes:**
1. **Input validation**:
   ```python
   if torch.isnan(audio).any() or torch.isnan(video).any():
       skip batch
   ```

2. **Loss validation**:
   ```python
   if torch.isnan(loss) or torch.isinf(loss):
       skip batch
   ```

3. **Gradient validation**:
   ```python
   if has_nan_grad:
       skip update
   ```

4. **Gradient norm check**:
   ```python
   if grad_norm > 100:
       skip update
   ```

**Benefits:**
- ✅ Prevents NaN propagation
- ✅ Skips problematic batches
- ✅ Continues training despite issues
- ✅ Reports NaN occurrences

---

## 🚀 HOW TO USE FIXED VERSION

### Option 1: Pull Latest Code
```bash
cd dry_watermelon
git pull origin main
```

### Option 2: Manual Update

#### Update Audio Extraction:
Replace `_extract_audio` method in `data/simple_ravdess_dataset.py` with new version.

#### Update Training Loop:
Replace `train_epoch` function in `colab_train_easy.py` with new version.

---

## 📋 INSTALLATION CHECKLIST FOR COLAB

### Step 1: Install FFmpeg Properly
```python
# In Colab
!apt-get update -qq
!apt-get install -y ffmpeg libavcodec-extra
!ffmpeg -version
```

**Verify:**
```
ffmpeg version 4.x.x or higher
```

### Step 2: Install Audio Libraries
```python
!pip install -q librosa soundfile
!pip install -q torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Test Audio Extraction
```python
import librosa
import soundfile as sf

# Test librosa
audio, sr = librosa.load("test.mp4", sr=16000)
print(f"✅ Librosa works: {audio.shape}")

# Test soundfile
audio, sr = sf.read("test.wav")
print(f"✅ Soundfile works: {audio.shape}")
```

---

## 🧪 TESTING FIXES

### Test 1: Audio Extraction
```python
from data.simple_ravdess_dataset import SimpleRAVDESSDataset
import torch

dataset = SimpleRAVDESSDataset(
    data_dir="/content/drive/MyDrive/RAVDESS",
    split="train",
    use_audio=True,
)

# Load one sample
audio, video, label, metadata = dataset[0]

print(f"Audio shape: {audio.shape}")
print(f"Audio min: {audio.min():.4f}, max: {audio.max():.4f}")
print(f"Audio mean: {audio.mean():.4f}, std: {audio.std():.4f}")

# Check if audio is not silent
if audio.abs().max() < 0.001:
    print("⚠️  WARNING: Audio is silent!")
else:
    print("✅ Audio extraction working!")
```

**Expected output:**
```
Audio shape: torch.Size([48000])
Audio min: -0.5234, max: 0.4567
Audio mean: 0.0012, std: 0.1234
✅ Audio extraction working!
```

### Test 2: Training Stability
```python
# Quick test with 1 epoch
!python colab_train_easy.py

# Check for NaN
# Should see:
# - No "loss=nan"
# - Accuracy > 0%
# - Loss decreasing
```

**Expected output:**
```
Epoch 1/50
Training: 100%|██████████| 120/120 [01:02<00:00, loss=1.8234, acc=35.23%, nan=0]
✅ No NaN losses
```

---

## 🔍 DEBUGGING TIPS

### If Audio Still Fails:

#### Check 1: FFmpeg Installation
```python
!which ffmpeg
!ffmpeg -version
```

#### Check 2: Video File Integrity
```python
import cv2

cap = cv2.VideoCapture("/path/to/video.mp4")
print(f"Opened: {cap.isOpened()}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
cap.release()
```

#### Check 3: Librosa
```python
import librosa

try:
    audio, sr = librosa.load("/path/to/video.mp4", sr=16000)
    print(f"✅ Librosa works: {audio.shape}")
except Exception as e:
    print(f"❌ Librosa failed: {e}")
```

### If Loss Still NaN:

#### Check 1: Learning Rate
```python
# Try lower learning rate
"learning_rate": 1e-5,  # Instead of 1e-4
```

#### Check 2: Batch Size
```python
# Try smaller batch size
"batch_size": 4,  # Instead of 8
```

#### Check 3: Model Initialization
```python
# Check model parameters
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"❌ NaN in {name}")
    if torch.isinf(param).any():
        print(f"❌ Inf in {name}")
```

#### Check 4: Data
```python
# Check one batch
for audio, video, labels, _ in train_loader:
    print(f"Audio: min={audio.min():.4f}, max={audio.max():.4f}")
    print(f"Video: min={video.min():.4f}, max={video.max():.4f}")
    print(f"Labels: {labels}")
    break
```

---

## 📊 EXPECTED BEHAVIOR AFTER FIXES

### Training Output:
```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|██████████| 120/120 [01:02<00:00, loss=1.8234, acc=35.23%, nan=0]
Validation: 100%|██████████| 24/24 [00:08<00:00]

Results:
  Train Loss: 1.8234, Acc: 35.23%
  Val Loss: 1.6543, Acc: 42.15%
  LR: 1.00e-04
  🎉 New best! Val Acc: 42.15%

Epoch 10/50
----------------------------------------------------------------------
Training: 100%|██████████| 120/120 [01:02<00:00, loss=0.8234, acc=68.45%, nan=0]
Validation: 100%|██████████| 24/24 [00:08<00:00]

Results:
  Train Loss: 0.8234, Acc: 68.45%
  Val Loss: 0.9123, Acc: 65.32%
  LR: 5.23e-05
```

### Key Indicators:
- ✅ Loss is a number (not NaN)
- ✅ Loss decreases over epochs
- ✅ Accuracy increases over epochs
- ✅ nan=0 (no NaN occurrences)
- ✅ Gradient norm < 100

---

## 🎯 QUICK FIX CHECKLIST

Before training:
- [ ] Pull latest code: `git pull origin main`
- [ ] Install FFmpeg: `!apt-get install -y ffmpeg`
- [ ] Install librosa: `!pip install librosa soundfile`
- [ ] Test audio extraction (see Test 1 above)
- [ ] Verify no NaN in first batch

During training:
- [ ] Monitor for "loss=nan"
- [ ] Check "nan=X" counter (should be 0)
- [ ] Verify accuracy > 0%
- [ ] Watch loss decreasing

If issues persist:
- [ ] Lower learning rate to 1e-5
- [ ] Reduce batch size to 4
- [ ] Disable audio: `"use_audio": False`
- [ ] Check data integrity

---

## 💡 ALTERNATIVE: DISABLE AUDIO

If audio extraction continues to fail:

```python
# In colab_train_easy.py CONFIG:
"use_audio": False,  # Train with video only
```

**Expected results with video-only:**
- Accuracy: ~70-75% (lower than multimodal)
- No audio extraction errors
- Faster training

---

## 📞 SUPPORT

If fixes don't work:
1. Check error messages carefully
2. Run debugging tests above
3. Try video-only mode
4. Report issue with full error log

---

**Last Updated**: January 29, 2026
**Status**: ✅ Fixes Applied
**Tested**: Pending user verification
