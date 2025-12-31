# 🎬 Inference Guide

## Tổng quan

Hướng dẫn sử dụng model đã train để predict emotion từ video mới.

---

## 🚀 Quick Start

### **Cài đặt ffmpeg** (Bắt buộc)

ffmpeg cần thiết để extract audio từ video.

#### Windows:
```bash
# Option 1: Chocolatey
choco install ffmpeg

# Option 2: Download từ https://ffmpeg.org/download.html
# Thêm vào PATH
```

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Mac:
```bash
brew install ffmpeg
```

**Verify installation:**
```bash
ffmpeg -version
```

---

## 📝 Usage

### **Basic Inference**

```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth
```

### **With All Probabilities**

```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth \
    --show-all-probs
```

### **On CPU**

```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth \
    --device cpu
```

---

## 📊 Example Output

```
============================================================
EMOTION RECOGNITION - INFERENCE
============================================================
Video: data/test_samples/01-02-01-01-01-01-01.mp4
Checkpoint: checkpoints/test_samples/best_model.pth
Device: cuda
============================================================

Loading model from: checkpoints/test_samples/best_model.pth
✓ Model loaded successfully
  Device: cuda

Processing video: data/test_samples/01-02-01-01-01-01-01.mp4
  [1/3] Extracting audio...
  [2/3] Extracting video frames...
  [3/3] Running inference...

============================================================
PREDICTION RESULT
============================================================

🎭 Predicted Emotion: NEUTRAL
   Confidence: 95.67%

📊 Top-3 Predictions:
   1. neutral    ████████████████████████████████████████ 95.67%
   2. calm       ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3.12%
   3. happy      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.89%

📈 All Probabilities:
   neutral   : 0.9567
   calm      : 0.0312
   happy     : 0.0089
   sad       : 0.0021
   angry     : 0.0007
   fearful   : 0.0003
   disgust   : 0.0001
   surprised : 0.0000
============================================================
```

---

## 🎯 Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | ✅ Yes | - | Path to video file (.mp4, .avi, etc.) |
| `--checkpoint` | ✅ Yes | - | Path to model checkpoint (.pth) |
| `--device` | ❌ No | cuda | Device (cuda/cpu) |
| `--show-all-probs` | ❌ No | False | Show all class probabilities |

---

## 📁 Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- Any format supported by ffmpeg

**Requirements:**
- Video duration: ~3 seconds (will be trimmed/padded)
- Audio: Will be resampled to 16kHz mono
- Video: Will be resized to 224x224

---

## 🔧 Troubleshooting

### **1. ffmpeg not found**

```
Error: [WinError 2] The system cannot find the file specified
```

**Solution:** Install ffmpeg (see above)

### **2. CUDA out of memory**

```
RuntimeError: CUDA out of memory
```

**Solution:** Use CPU
```bash
python scripts/inference.py --video video.mp4 --checkpoint model.pth --device cpu
```

### **3. No frames extracted**

```
Warning: No frames extracted, using blank frames
```

**Solution:** Check video file is valid
```bash
ffmpeg -i video.mp4
```

### **4. Checkpoint not found**

```
Error: Checkpoint not found: checkpoints/best_model.pth
```

**Solution:** Train model first
```bash
python scripts/train_test_samples.py
```

---

## 💡 Tips

### **Batch Inference**

Process multiple videos:

```bash
# Create a script
for video in data/test_samples/*.mp4; do
    echo "Processing: $video"
    python scripts/inference.py \
        --video "$video" \
        --checkpoint checkpoints/test_samples/best_model.pth
done
```

### **Save Results to File**

```bash
python scripts/inference.py \
    --video video.mp4 \
    --checkpoint model.pth \
    --show-all-probs > results.txt
```

### **Python API**

Use in your own code:

```python
from scripts.inference import VideoEmotionPredictor

# Create predictor
predictor = VideoEmotionPredictor(
    checkpoint_path="checkpoints/best_model.pth",
    device="cuda"
)

# Predict
result = predictor.predict("video.mp4")

# Get emotion
emotion = result["predicted_emotion"]
confidence = result["confidence"]

print(f"Emotion: {emotion} ({confidence:.2%})")
```

---

## 📊 Understanding Results

### **Confidence Score**

- **> 90%**: Very confident, reliable prediction
- **70-90%**: Confident, good prediction
- **50-70%**: Moderate confidence
- **< 50%**: Low confidence, uncertain

### **Top-3 Predictions**

Shows the 3 most likely emotions. Useful when:
- Confidence is low
- Multiple emotions are present
- Ambiguous expressions

### **All Probabilities**

Shows probability for each emotion class. Useful for:
- Understanding model uncertainty
- Detecting mixed emotions
- Debugging model behavior

---

## 🎬 Example Videos

### **Test with sample videos:**

```bash
# Neutral emotion
python scripts/inference.py \
    --video data/test_samples/01-02-01-01-01-01-01.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth

# Calm emotion
python scripts/inference.py \
    --video data/test_samples/01-02-01-01-01-02-01.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth

# Happy emotion
python scripts/inference.py \
    --video data/test_samples/01-02-01-01-02-01-01.mp4 \
    --checkpoint checkpoints/test_samples/best_model.pth
```

---

## 📚 Next Steps

1. ✅ Install ffmpeg
2. ✅ Train model (if not done)
3. ✅ Run inference on test videos
4. ⏳ Try on your own videos
5. ⏳ Integrate into your application

---

## 🎯 Summary

**Command:**
```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --show-all-probs
```

**Output:**
- Predicted emotion
- Confidence score
- Top-3 predictions
- All probabilities (optional)

**Requirements:**
- ffmpeg installed
- Trained model checkpoint
- Video file

**Ready to predict emotions! 🎭**
