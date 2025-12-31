# 🎓 Training Guide - Multimodal FER với LFM2

## 📋 Tổng quan

Hướng dẫn chi tiết về cách train và finetune mô hình Multimodal FER sử dụng Liquid LFM2-700M làm fusion backbone.

---

## 🎯 Chiến lược Training

### **Giai đoạn 1: Pretrain từng Branch riêng lẻ** (Khuyến nghị)

Trước khi train toàn bộ model, nên pretrain từng branch riêng:

#### 1.1. Audio Branch Pretraining
```python
# Train audio branch với audio-only emotion recognition
audio_branch = AudioBranch(
    pretrained_model=None,  # Hoặc dùng pretrained FastConformer
    feature_dim=512,
    num_layers=4,  # Lightweight cho RTX 3050
    num_segments=8,
)

# Loss: CrossEntropy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Lợi ích:**
- Audio branch học được acoustic features tốt
- Giảm thời gian train toàn bộ model
- Có thể dùng pretrained FastConformer từ NVIDIA NeMo

#### 1.2. Visual Branch Pretraining
```python
# Train visual branch với video-only emotion recognition
visual_branch = VisualBranch(
    pretrained_model="google/siglip2-base-patch16-224",
    freeze_encoder=True,  # Freeze SigLIP, chỉ train ROI + Temporal
    num_keep_tokens=64,
    temporal_depth=6,
)

# Loss: CrossEntropy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Lợi ích:**
- Visual branch học được facial expressions
- SigLIP đã pretrained trên large-scale vision data
- Chỉ cần train ROI compression và temporal encoder

---

### **Giai đoạn 2: Multimodal Fusion Training**

Sau khi có pretrained branches, train toàn bộ model:

#### 2.1. Load Pretrained Branches
```python
# Load pretrained weights
audio_branch.load_state_dict(torch.load("audio_branch_pretrained.pth"))
visual_branch.load_state_dict(torch.load("visual_branch_pretrained.pth"))

# Freeze branches (optional)
for param in audio_branch.parameters():
    param.requires_grad = False
for param in visual_branch.parameters():
    param.requires_grad = False
```

#### 2.2. Train Fusion + Classifier
```python
model = MultimodalFER(
    audio_config=audio_config,
    visual_config=visual_config,
    fusion_config=LFM2FusionConfig(
        pretrained_model="LiquidAI/LFM2-700M",
        use_pretrained=True,
        freeze_backbone=False,  # Finetune LFM2
        num_layers=6,  # Dùng 6 layers đầu của LFM2
    ),
)
```

---

## 🔥 Loss Functions

### **1. Primary Loss: CrossEntropy với Label Smoothing**

```python
class EmotionLoss(nn.Module):
    def __init__(
        self,
        num_classes=8,
        label_smoothing=0.1,
        class_weights=None,
    ):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )
    
    def forward(self, logits, labels):
        return self.ce_loss(logits, labels)
```

**Tại sao dùng Label Smoothing?**
- Giảm overfitting
- Model không quá confident vào 1 class
- Cải thiện generalization

**Class Weights:**
```python
# Nếu dataset imbalanced
class_weights = torch.tensor([
    1.0,  # neutral
    1.2,  # calm (ít sample hơn)
    1.0,  # happy
    1.1,  # sad
    1.0,  # angry
    1.3,  # fearful (ít sample hơn)
    1.2,  # disgust (ít sample hơn)
    1.0,  # surprised
])
```

---

### **2. Auxiliary Loss: Modality-Specific Losses** (Optional)

Để đảm bảo mỗi modality học tốt:

```python
class MultimodalLoss(nn.Module):
    def __init__(
        self,
        num_classes=8,
        alpha_audio=0.3,
        alpha_visual=0.3,
        alpha_fusion=1.0,
    ):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.alpha_audio = alpha_audio
        self.alpha_visual = alpha_visual
        self.alpha_fusion = alpha_fusion
        
        # Auxiliary classifiers
        self.audio_classifier = nn.Linear(512, num_classes)
        self.visual_classifier = nn.Linear(768, num_classes)
    
    def forward(self, outputs, labels):
        # Main fusion loss
        loss_fusion = self.ce_loss(outputs["logits"], labels)
        
        # Auxiliary losses
        if "audio_features" in outputs:
            audio_logits = self.audio_classifier(
                outputs["audio_features"].mean(dim=1)
            )
            loss_audio = self.ce_loss(audio_logits, labels)
        else:
            loss_audio = 0
        
        if "visual_features" in outputs:
            visual_logits = self.visual_classifier(
                outputs["visual_features"].mean(dim=1)
            )
            loss_visual = self.ce_loss(visual_logits, labels)
        else:
            loss_visual = 0
        
        # Total loss
        total_loss = (
            self.alpha_fusion * loss_fusion +
            self.alpha_audio * loss_audio +
            self.alpha_visual * loss_visual
        )
        
        return {
            "loss": total_loss,
            "loss_fusion": loss_fusion,
            "loss_audio": loss_audio,
            "loss_visual": loss_visual,
        }
```

---

### **3. Contrastive Loss: Audio-Visual Alignment** (Advanced)

Để đảm bảo audio và visual features aligned:

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, audio_features, visual_features):
        # Normalize features
        audio_norm = F.normalize(audio_features, dim=-1)
        visual_norm = F.normalize(visual_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_norm, visual_norm.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(audio_features.size(0)).to(audio_features.device)
        
        # Contrastive loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
```

---

## 🎛️ Hyperparameters

### **Optimizer: AdamW**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)
```

**Learning Rates khác nhau cho từng component:**

```python
# Differential learning rates
param_groups = [
    # Pretrained branches: lower LR
    {"params": model.audio_branch.parameters(), "lr": 1e-5},
    {"params": model.visual_branch.parameters(), "lr": 1e-5},
    
    # Fusion: medium LR
    {"params": model.fusion.parameters(), "lr": 5e-5},
    
    # Classifier: higher LR
    {"params": model.classifier.parameters(), "lr": 1e-4},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

---

### **Learning Rate Scheduler**

#### Option 1: Cosine Annealing with Warmup
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6,
)
```

#### Option 2: ReduceLROnPlateau
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True,
)
```

---

### **Regularization**

#### 1. Dropout
```python
# Đã có trong model
dropout = 0.1  # Standard
```

#### 2. Mixup (Data Augmentation)
```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 3. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📊 Training Configuration

### **Cho RTX 3050 (12GB VRAM)**

```yaml
training:
  # Basic
  batch_size: 4  # Nhỏ để fit memory
  gradient_accumulation_steps: 4  # Effective batch_size = 16
  max_epochs: 100
  
  # Optimizer
  optimizer: AdamW
  lr: 1e-4
  weight_decay: 0.01
  
  # Scheduler
  scheduler: CosineAnnealingWarmRestarts
  warmup_epochs: 5
  
  # Loss
  loss: CrossEntropyLoss
  label_smoothing: 0.1
  
  # Regularization
  dropout: 0.1
  gradient_clip: 1.0
  mixup_alpha: 0.2
  
  # Memory optimization
  mixed_precision: true  # FP16
  gradient_checkpointing: true
```

---

## 🔄 Training Loop Example

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Setup
model = MultimodalFER(...)
criterion = EmotionLoss(num_classes=8, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # For mixed precision

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (audio, video, labels) in enumerate(train_loader):
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward
        with autocast():
            outputs = model(audio, video)
            loss = criterion(outputs["logits"], labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

---

## 📈 Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, video, labels in val_loader:
            audio = audio.to(device)
            video = video.to(device)
            
            outputs = model(audio, video)
            preds = outputs["probabilities"].argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }
```

---

## 💡 Tips & Best Practices

### **1. Curriculum Learning**
Bắt đầu với easy samples, dần dần tăng độ khó:
```python
# Sort dataset by difficulty (e.g., confidence score)
# Train on easy samples first, then add harder samples
```

### **2. Data Augmentation**
- **Audio**: Time stretch, pitch shift, noise injection
- **Video**: Random crop, horizontal flip, color jitter
- **Both**: Mixup, CutMix

### **3. Early Stopping**
```python
early_stopping = EarlyStopping(
    patience=15,
    min_delta=0.001,
    mode='max',  # For accuracy
)
```

### **4. Model Checkpointing**
```python
# Save best model
if val_accuracy > best_accuracy:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': val_accuracy,
    }, 'best_model.pth')
```

### **5. Ablation Studies**
Test từng component:
- Audio only
- Visual only
- Audio + Visual (no fusion)
- Full model

---

## 🎯 Expected Results

### **RAVDESS Dataset**

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Audio Only | ~65-70% | ~0.63 |
| Visual Only | ~70-75% | ~0.68 |
| Multimodal (Early Fusion) | ~75-80% | ~0.73 |
| **Multimodal (LFM2 Fusion)** | **~80-85%** | **~0.78** |

---

## 🚀 Next Steps

1. **Implement training pipeline** (`training/trainer.py`)
2. **Create dataset loaders** (`data/datasets/ravdess.py`)
3. **Add logging** (TensorBoard/WandB)
4. **Hyperparameter tuning** (Optuna/Ray Tune)
5. **Extended datasets** (CREMA-D, DFEW, MELD)

---

## 📚 References

- **LFM2**: [Liquid Foundation Models](https://www.liquid.ai/)
- **Label Smoothing**: [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567)
- **Mixup**: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- **Contrastive Learning**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
