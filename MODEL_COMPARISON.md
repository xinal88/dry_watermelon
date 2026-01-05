# 📊 Model Architecture Comparison

So sánh giữa model gốc và model tối ưu cho RTX 3050.

## 🏗️ Architecture Comparison

### Audio Branch (Conformer Encoder)

| Component | Original | RTX 3050 Optimized | Reduction |
|-----------|----------|-------------------|-----------|
| Feature Dim | 512 | **128** | **75% ↓** |
| Num Layers | 8 | **4** | **50% ↓** |
| Attention Heads | 8 | 4 | 50% ↓ |
| FFN Dim | 2048 | 512 | 75% ↓ |
| **Params** | **~50M** | **~9M** | **82% ↓** |

### Visual Branch

| Component | Original | RTX 3050 Optimized | Reduction |
|-----------|----------|-------------------|-----------|
| Feature Dim | 512 | **128** | **75% ↓** |
| Temporal Depth | 4 | **2** | **50% ↓** |
| GSCB Layers | 2 | 1 | 50% ↓ |
| Attention Layers | 2 | 1 | 50% ↓ |
| **Params** | **~14M** | **~1M** | **93% ↓** |

### LFM2 Fusion

| Component | Original | RTX 3050 Optimized | Reduction |
|-----------|----------|-------------------|-----------|
| Audio Input | 512 | **128** | 75% ↓ |
| Visual Input | 512 | **128** | 75% ↓ |
| Hidden Dim | 1024 | **256** | **75% ↓** |
| Num Layers | 4 | **2** | **50% ↓** |
| Output Dim | 512 | **128** | **75% ↓** |
| **Params** | **~84M** | **~4M** | **95% ↓** |

### Classifier Head

| Component | Original | RTX 3050 Optimized | Reduction |
|-----------|----------|-------------------|-----------|
| Input Dim | 512 | **128** | 75% ↓ |
| Hidden Layers | [512, 256] | **[128, 64]** | 75% ↓ |
| **Params** | **~0.4M** | **~0.03M** | **92% ↓** |

## 📈 Overall Model Statistics

| Metric | Original | RTX 3050 Optimized | Reduction |
|--------|----------|-------------------|-----------|
| **Total Parameters** | **149M** | **~14M** | **90% ↓** |
| **FP32 Memory** | 0.60 GB | **0.06 GB** | **90% ↓** |
| **FP16 Memory** | 0.30 GB | **0.03 GB** | **90% ↓** |
| **Training VRAM** | ~3.8 GB | **~2.5 GB** | **34% ↓** |

## 🎯 Training Configuration

### Dataset

| Setting | Original | RTX 3050 Optimized | Change |
|---------|----------|-------------------|--------|
| Training Samples | 1920 | **960** | **50% ↓** |
| Val Samples | 480 | 480 | Same |
| Test Samples | 480 | 480 | Same |
| Batch Size | 8-16 | **2** | **75-87% ↓** |

### Training Hyperparameters

| Setting | Original | RTX 3050 Optimized | Change |
|---------|----------|-------------------|--------|
| Epochs | 100 | **50** | 50% ↓ |
| Learning Rate | 1e-4 | 1e-4 | Same |
| Mixed Precision | FP16 | FP16 | Same |
| Gradient Clip | 1.0 | 1.0 | Same |

## ⏱️ Time Estimation

### Per Epoch Timing

```
Training samples: 960
Batch size: 2
Batches per epoch: 480

Estimated time per batch: ~3-4 seconds
Estimated time per epoch: 480 × 3.5s = 1680s ≈ 28 minutes
```

### Total Training Time

```
Total epochs: 50
Time per epoch: ~28 minutes

Total training time: 50 × 28 = 1400 minutes ≈ 23 hours
```

**⚠️ Lưu ý:** Đây là ước tính conservative. Thực tế có thể nhanh hơn:
- Epoch đầu tiên chậm hơn (loading, compilation)
- Các epoch sau nhanh hơn (~20-25 phút/epoch)
- **Ước tính thực tế: 18-20 giờ**

### Breakdown by Phase

| Phase | Time per Epoch | Total Time (50 epochs) |
|-------|---------------|----------------------|
| Data Loading | ~2 min | ~1.7 hours |
| Forward Pass | ~15 min | ~12.5 hours |
| Backward Pass | ~8 min | ~6.7 hours |
| Validation | ~3 min | ~2.5 hours |
| **Total** | **~28 min** | **~23 hours** |

## 🎯 Expected Performance

### Accuracy Comparison

| Metric | Original (Full) | RTX 3050 (Half) | Difference |
|--------|----------------|-----------------|------------|
| **UAR** | 0.65-0.75 | **0.50-0.60** | -0.10-0.15 |
| **Accuracy** | 0.70-0.80 | **0.55-0.65** | -0.10-0.15 |
| **WAR** | 0.68-0.78 | **0.53-0.63** | -0.10-0.15 |

**Lý do performance thấp hơn:**
1. ✂️ Model nhỏ hơn 90% (14M vs 149M params)
2. 📊 Chỉ dùng 50% training data (960 vs 1920 samples)
3. 🔢 Batch size nhỏ (2 vs 8-16) - ảnh hưởng batch normalization
4. ⏱️ Ít epochs hơn (50 vs 100)

## 💡 Trade-offs

### Advantages ✅

1. **Fits in 4GB VRAM** - Chắc chắn không OOM
2. **Faster per epoch** - Ít computation hơn
3. **Less overfitting risk** - Model nhỏ hơn
4. **Can train locally** - Không cần Colab

### Disadvantages ❌

1. **Lower accuracy** - ~10-15% UAR drop
2. **Longer total time** - 18-20 giờ vs 3-4 giờ (T4)
3. **Less capacity** - Khó học patterns phức tạp
4. **Smaller batch** - Training kém stable hơn

## 🔄 Optimization Strategy

### Những gì đã giảm (theo thứ tự ưu tiên):

1. **Feature dimensions** (512→128): Giảm 75%
   - Ảnh hưởng lớn nhất đến params
   - Trade-off: Mất capacity

2. **Number of layers** (8→4, 4→2): Giảm 50%
   - Giảm depth, giữ width
   - Trade-off: Mất khả năng học hierarchical features

3. **Hidden dimensions** (1024→256): Giảm 75%
   - Trong fusion layers
   - Trade-off: Bottleneck trong fusion

4. **Training data** (1920→960): Giảm 50%
   - Nhanh hơn, ít overfitting
   - Trade-off: Ít data để học

5. **Batch size** (8→2): Giảm 75%
   - Cần thiết cho VRAM
   - Trade-off: Noisy gradients

### Những gì giữ nguyên:

✅ **Architecture design** - Vẫn giữ cấu trúc multimodal  
✅ **Attention mechanisms** - Vẫn có self-attention  
✅ **Fusion strategy** - Vẫn dùng LFM2  
✅ **Learning rate** - Không thay đổi  
✅ **Validation/Test sets** - Full data để đánh giá đúng  

## 📊 Memory Breakdown (During Training)

### Original Model
```
Model weights:        0.60 GB (FP32) / 0.30 GB (FP16)
Activations:          1.50 GB
Gradients:            0.60 GB
Optimizer states:     1.20 GB
Batch data:           0.50 GB
Total:                ~3.8 GB
```

### RTX 3050 Optimized
```
Model weights:        0.06 GB (FP32) / 0.03 GB (FP16)
Activations:          0.80 GB (smaller batch + model)
Gradients:            0.06 GB
Optimizer states:     0.12 GB
Batch data:           0.25 GB (batch_size=2)
Buffer:               1.24 GB (safety margin)
Total:                ~2.5 GB (safe for 4GB)
```

## 🎓 Recommendations

### Nếu bạn có thời gian (18-20 giờ):
✅ **Chạy script này** - An toàn, chắc chắn hoàn thành

### Nếu bạn cần kết quả nhanh hơn:
1. **Giảm epochs xuống 30** - Tiết kiệm 40% thời gian
2. **Dùng Colab T4** - Nhanh hơn 5-6x
3. **Train overnight** - Để máy chạy qua đêm

### Nếu bạn cần accuracy cao hơn:
1. **Tăng feature dim lên 192** - Compromise giữa size và accuracy
2. **Dùng full dataset** - Bỏ random 50%
3. **Train longer** - 100 epochs thay vì 50

## 🚀 Quick Commands

### Start training (18-20 hours)
```bash
python scripts/train_half_dataset.py
```

### Monitor progress
```bash
# In another terminal
watch -n 5 nvidia-smi
```

### Check results
```bash
cat checkpoints/half_dataset_rtx3050/history.json
```

### Resume if interrupted
```bash
python scripts/resume_training.py checkpoints/half_dataset_rtx3050/checkpoint_epoch_20.pt
```

---

**Bottom line:** Model đã giảm 90% params nhưng vẫn giữ được cấu trúc multimodal. Thời gian train ~18-20 giờ, accuracy dự kiến 0.50-0.60 UAR (acceptable cho demo/testing).
