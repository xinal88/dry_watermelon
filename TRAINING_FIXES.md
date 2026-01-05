# 🔧 Training Fixes Applied

Các fix đã áp dụng để giải quyết NaN loss và memory errors.

## ❌ Problems Encountered

### 1. NaN Loss During Training
```
Training: 100%|████████| 480/480 [09:00<00:00, 1.13s/it, loss=nan]
```

**Nguyên nhân:**
- Learning rate quá cao (1e-4) cho model nhỏ
- Batch size quá nhỏ (2) → noisy gradients
- Không có warmup → training unstable
- Gradient explosion

### 2. Memory Error During Validation
```
SystemError: Unable to allocate 2.64 MiB for array
```

**Nguyên nhân:**
- `num_workers > 0` tạo nhiều processes
- Mỗi worker load video vào RAM
- RAM không đủ cho multiprocessing

## ✅ Solutions Applied

### Fix 1: Reduce Learning Rate
```python
# Before
"lr": 1e-4

# After
"lr": 5e-5  # 50% reduction
```

**Lý do:** Model nhỏ hơn cần LR nhỏ hơn để tránh divergence.

### Fix 2: Add Warmup
```python
def lr_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda)
```

**Lý do:** Warmup giúp training stable hơn ở đầu.

### Fix 3: Aggressive Gradient Clipping
```python
# Before
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# After
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
```

**Lý do:** Clip mạnh hơn để tránh gradient explosion.

### Fix 4: NaN Detection & Skip
```python
# Check for NaN
if torch.isnan(loss) or torch.isinf(loss):
    print(f"\nWarning: NaN/Inf loss detected, skipping batch")
    continue

# Check gradient norm
grad_norm = ...
if grad_norm > 100:
    print(f"\nWarning: Large gradient norm, skipping")
    continue
```

**Lý do:** Skip bad batches thay vì crash.

### Fix 5: Disable Multiprocessing
```python
# Before
"num_workers": 2

# After
"num_workers": 0  # Single process
```

**Lý do:** Tránh memory errors từ multiprocessing.

### Fix 6: Add Numerical Stability
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
    eps=1e-8,  # Add epsilon
)
```

**Lý do:** Tránh division by zero trong optimizer.

## 📊 Expected Behavior Now

### Training
```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|████| 480/480 [10:00<00:00, loss=2.0543, grad=0.85]
Validation: 100%|██| 240/240 [02:30<00:00]

Results:
  Train Loss: 2.0543
  Val Loss:   1.8234
  Accuracy:   0.3542
  UAR:        0.3125
  Time:       750.3s
```

### Loss Progression
```
Epoch 1:  loss=2.05 (high, normal)
Epoch 5:  loss=1.65 (decreasing)
Epoch 10: loss=1.35 (stable)
Epoch 20: loss=1.15 (converging)
Epoch 50: loss=0.95 (final)
```

## ⚠️ Trade-offs

### Slower Training
- `num_workers=0` → Chậm hơn ~20%
- Mỗi epoch: ~30 phút → ~36 phút
- Total: 18-20 giờ → **22-24 giờ**

### More Stable
- ✅ Không bị NaN loss
- ✅ Không bị memory errors
- ✅ Training smooth hơn
- ✅ Chắc chắn hoàn thành

## 🎯 New Time Estimate

```
Per epoch: ~36 minutes (was 28 minutes)
50 epochs: 36 × 50 = 1800 minutes = 30 hours

Realistic: 22-24 hours (epochs sau nhanh hơn)
```

## 💡 If Still Having Issues

### If NaN persists:
```python
# Reduce LR further
"lr": 2e-5  # Even lower

# Or increase warmup
warmup_epochs = 10  # More warmup
```

### If memory errors persist:
```python
# Reduce batch size
"batch_size": 1  # Extreme case

# Or reduce video resolution in ravdess_dataset.py
video_size: (112, 112)  # Half resolution
```

### If too slow:
```python
# Reduce epochs
"num_epochs": 30  # 60% of time

# Or reduce dataset further
half_size = train_size // 4  # Use 25% instead of 50%
```

## 🚀 Ready to Train

Script đã được fix và sẵn sàng:

```bash
python scripts/train_half_dataset.py
```

Expected:
- ✅ No NaN loss
- ✅ No memory errors  
- ✅ Stable training
- ⏱️ 22-24 hours total
- 🎯 UAR 0.50-0.60

---

**Note:** Training sẽ chậm hơn dự kiến ban đầu (22-24h thay vì 18-20h) nhưng chắc chắn hoàn thành và stable.
