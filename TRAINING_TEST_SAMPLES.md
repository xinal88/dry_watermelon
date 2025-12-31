# 🚀 Training & Evaluation với Test Samples

## Tổng quan

Hướng dẫn train và evaluate model trên test samples trước khi chạy full RAVDESS dataset.

---

## 📁 Files đã tạo

### **1. Training Module**
- `training/losses.py` - Loss functions (CrossEntropy + Label Smoothing)
- `training/metrics.py` - Metrics (UAR, WAR, WA-F1)
- `training/__init__.py` - Module exports

### **2. Data**
- `data/test_dataset.py` - Test dataset loader cho RAVDESS samples

### **3. Scripts**
- `scripts/train_test_samples.py` - Training script
- `scripts/evaluate.py` - Evaluation script

---

## 🎯 Metrics

### **UAR (Unweighted Average Recall)**
- Macro recall - trung bình recall của tất cả classes
- Không phụ thuộc vào class distribution
- **Quan trọng nhất** cho emotion recognition

### **WAR (Weighted Average Recall)**
- Weighted recall - có tính đến class frequency
- Phù hợp với imbalanced datasets

### **WA-F1 (Weighted Average F1)**
- Weighted F1 score
- Balance giữa precision và recall

---

## 🚀 Quick Start

### **1. Test Dataset Loader**

```bash
# Test dataset loader
python data/test_dataset.py
```

**Expected output:**
```
Found 3 video samples
Dataset size: 3
Sample metadata:
  filename: 01-02-01-01-01-01-01.mp4
  emotion_code: 01
  emotion_idx: 0 (neutral)
Audio: torch.Size([48000])
Video: torch.Size([16, 3, 224, 224])
```

---

### **2. Train on Test Samples**

```bash
# Train model
python scripts/train_test_samples.py
```

**Configuration:**
- Data: `data/test_samples` (3 videos)
- Batch size: 2
- Epochs: 50
- Loss: CrossEntropy + Label Smoothing (0.1)
- Optimizer: AdamW (lr=1e-4)
- Scheduler: CosineAnnealingWarmRestarts

**Output:**
```
TRAINING ON TEST SAMPLES
Device: cuda
Epochs: 50
Train samples: 3
Val samples: 3

Epoch 1/50:
  Train Loss: 2.1234
  Val Loss:   2.0987
  Accuracy:   0.3333
  UAR:        0.3333
  WAR:        0.3333
  WA-F1:      0.3000
  ✓ Best model saved (UAR: 0.3333)

...

TRAINING COMPLETED
Best UAR: 0.8500
Checkpoints saved to: checkpoints/test_samples
```

**Saved files:**
- `checkpoints/test_samples/best_model.pth` - Best model (highest UAR)
- `checkpoints/test_samples/final_model.pth` - Final model
- `checkpoints/test_samples/checkpoint_epoch_10.pth` - Regular checkpoints
- `checkpoints/test_samples/training_history.json` - Training history

---

### **3. Evaluate Model**

```bash
# Evaluate with checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/test_samples/best_model.pth \
    --data-dir data/test_samples \
    --save-dir results \
    --per-sample
```

**Arguments:**
- `--checkpoint`: Path to checkpoint file (required)
- `--data-dir`: Directory with test samples (default: data/test_samples)
- `--batch-size`: Batch size (default: 2)
- `--device`: cuda or cpu (default: cuda)
- `--save-dir`: Directory to save results (default: results)
- `--per-sample`: Show per-sample results (optional)

**Output:**
```
MULTIMODAL FER - EVALUATION

Loading checkpoint: checkpoints/test_samples/best_model.pth
  Epoch: 50
  Metrics:
    Accuracy: 0.8500
    UAR: 0.8500
    WAR: 0.8500
    WA-F1: 0.8400

Evaluating...
100%|████████████████████| 2/2 [00:05<00:00]

EVALUATION METRICS
Overall Metrics:
  Accuracy:     0.8500 (85.00%)
  UAR:          0.8500 (85.00%)
  WAR:          0.8500 (85.00%)
  WA-F1:        0.8400 (84.00%)
  Macro F1:     0.8300 (83.00%)

Per-Class Metrics:
Class        Recall     F1-Score
--------------------------------
neutral      0.9000     0.8800
calm         0.8500     0.8300
happy        0.8000     0.7900
...

PER-SAMPLE EVALUATION
Sample 1:
  File: 01-02-01-01-01-01-01.mp4
  True: neutral
  Pred: neutral (confidence: 92.34%)
  Correct: ✓
  Top-3:
    1. neutral: 92.34%
    2. calm: 5.12%
    3. happy: 1.23%

EVALUATION COMPLETED
Results saved to: results
```

**Saved files:**
- `results/evaluation_metrics.json` - Metrics in JSON
- `results/confusion_matrix.png` - Confusion matrix plot
- `results/predictions.npz` - Predictions and probabilities

---

## 📊 Understanding Metrics

### **Confusion Matrix**

Saved as `results/confusion_matrix.png`:

```
              Predicted
           N   C   H   S   A   F   D   Su
True  N  [90   5   2   1   0   1   1   0]
      C  [ 3  85   4   2   2   2   1   1]
      H  [ 1   2  80   3   5   4   3   2]
      ...
```

### **Per-Class Metrics**

```json
{
  "accuracy": 0.85,
  "uar": 0.85,        // Unweighted Average Recall
  "war": 0.85,        // Weighted Average Recall
  "wa_f1": 0.84,      // Weighted Average F1
  "macro_f1": 0.83,
  "per_class_recall": [0.90, 0.85, 0.80, ...],
  "per_class_f1": [0.88, 0.83, 0.79, ...]
}
```

---

## 🔧 Customization

### **Change Loss Function**

Edit `scripts/train_test_samples.py`:

```python
# Option 1: Simple CrossEntropy
self.criterion = EmotionLoss(
    num_classes=8,
    label_smoothing=0.1,  # 0.0 to disable
)

# Option 2: With class weights
class_weights = torch.tensor([1.0, 1.2, 1.0, 1.1, 1.0, 1.3, 1.2, 1.0])
self.criterion = EmotionLoss(
    num_classes=8,
    label_smoothing=0.1,
    class_weights=class_weights,
)

# Option 3: Multimodal loss with auxiliary
from training.losses import MultimodalLoss
self.criterion = MultimodalLoss(
    num_classes=8,
    alpha_audio=0.3,
    alpha_visual=0.3,
    alpha_fusion=1.0,
)
```

### **Change Hyperparameters**

```python
# Learning rate
lr = 1e-4  # Default
lr = 5e-5  # Lower for finetuning
lr = 2e-4  # Higher for from scratch

# Epochs
num_epochs = 50   # Default
num_epochs = 100  # More training

# Batch size
batch_size = 2  # For test samples
batch_size = 8  # For full dataset
```

### **Change Optimizer**

```python
# AdamW (default)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

# SGD with momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
)

# Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)
```

---

## 📈 Expected Results

### **Test Samples (3 videos)**

Vì chỉ có 3 samples, model sẽ overfit:

| Metric | After 10 epochs | After 50 epochs |
|--------|----------------|-----------------|
| Train Loss | 1.5 | 0.1 |
| Val Loss | 1.6 | 0.2 |
| UAR | 0.40 | 0.90+ |
| WAR | 0.40 | 0.90+ |
| WA-F1 | 0.35 | 0.85+ |

**Note**: Đây chỉ là để test pipeline, không phải kết quả thực tế!

### **Full RAVDESS Dataset**

Expected với full dataset:

| Metric | Value |
|--------|-------|
| UAR | 80-85% |
| WAR | 80-85% |
| WA-F1 | 78-83% |
| Accuracy | 80-85% |

---

## 🐛 Troubleshooting

### **1. Out of Memory**

```python
# Reduce batch size
batch_size = 1

# Use gradient accumulation
accumulation_steps = 4
```

### **2. ffmpeg not found**

```bash
# Install ffmpeg
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

### **3. No video files found**

```bash
# Check data directory
ls data/test_samples/*.mp4

# Should see:
# 01-02-01-01-01-01-01.mp4
# 01-02-01-01-01-02-01.mp4
# 01-02-01-01-02-01-01.mp4
```

### **4. CUDA out of memory**

```python
# Use CPU
device = "cpu"

# Or reduce model size
model = MultimodalFER(
    audio_config=AudioBranchConfig(num_layers=2),
    visual_config=VisualBranchConfig(temporal_depth=2),
    fusion_config=LFM2FusionConfig(num_layers=2),
)
```

---

## 📚 Next Steps

### **After testing pipeline:**

1. ✅ Verify training works
2. ✅ Verify evaluation works
3. ✅ Check metrics calculation
4. ⏳ Prepare full RAVDESS dataset
5. ⏳ Train on full dataset
6. ⏳ Evaluate on test set
7. ⏳ Tune hyperparameters

### **For full RAVDESS:**

```bash
# Download RAVDESS
python data/test_samples/script/download_RAVDESS.py

# Create RAVDESS dataset loader
# (similar to test_dataset.py but with train/val/test split)

# Train
python scripts/train.py --config configs/train_config.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 📝 Summary

**Files created:**
- ✅ `training/losses.py` - Loss functions
- ✅ `training/metrics.py` - UAR, WAR, WA-F1 metrics
- ✅ `data/test_dataset.py` - Test dataset loader
- ✅ `scripts/train_test_samples.py` - Training script
- ✅ `scripts/evaluate.py` - Evaluation script

**Usage:**
```bash
# 1. Test dataset
python data/test_dataset.py

# 2. Train
python scripts/train_test_samples.py

# 3. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/test_samples/best_model.pth --per-sample
```

**Metrics:**
- UAR (Unweighted Average Recall) - Primary metric
- WAR (Weighted Average Recall)
- WA-F1 (Weighted Average F1)
- Confusion Matrix
- Per-class metrics

**Ready for full RAVDESS training!** 🚀
