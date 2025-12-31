# 🚀 Quick Reference - Multimodal FER

## Model Creation

```python
from models import MultimodalFER

# Default configuration
model = MultimodalFER(num_classes=8, num_segments=8)

# Custom configuration
from models import (
    MultimodalFERConfig,
    AudioBranchConfig,
    VisualBranchConfig,
    LFM2FusionConfig,
    ClassifierConfig,
)

config = MultimodalFERConfig(
    audio_config=AudioBranchConfig(feature_dim=512, num_layers=4),
    visual_config=VisualBranchConfig(feature_dim=768, temporal_depth=6),
    fusion_config=LFM2FusionConfig(use_pretrained=True, num_layers=6),
    classifier_config=ClassifierConfig(hidden_dims=[512, 256]),
)

model = MultimodalFER.from_config(config)
```

## Forward Pass

```python
# Inputs
audio = torch.randn(batch_size, 48000)  # 3s at 16kHz
video = torch.randn(batch_size, 16, 3, 224, 224)  # 16 frames

# Forward
outputs = model(audio, video)

# Outputs
logits = outputs["logits"]  # [B, 8]
probs = outputs["probabilities"]  # [B, 8]
```

## Training

```python
# Setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# Training loop
model.train()
for audio, video, labels in train_loader:
    with torch.cuda.amp.autocast():
        outputs = model(audio, video)
        loss = criterion(outputs["logits"], labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Evaluation

```python
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for audio, video, labels in val_loader:
        outputs = model(audio, video)
        preds = outputs["probabilities"].argmax(dim=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(labels.cpu())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
```

## Loss Functions

```python
# 1. CrossEntropy + Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 2. With Class Weights
weights = torch.tensor([1.0, 1.2, 1.0, 1.1, 1.0, 1.3, 1.2, 1.0])
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# 3. Auxiliary Losses
loss_total = (
    1.0 * loss_fusion +
    0.3 * loss_audio +
    0.3 * loss_visual
)
```

## Hyperparameters

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)

# Differential LR
param_groups = [
    {"params": model.audio_branch.parameters(), "lr": 1e-5},
    {"params": model.visual_branch.parameters(), "lr": 1e-5},
    {"params": model.fusion.parameters(), "lr": 5e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4},
]
```

## Model Info

```python
# Print summary
model.print_summary()

# Count parameters
params = model.count_parameters()
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")

# Get emotion labels
labels = model.get_emotion_labels()
# ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
```

## Save/Load

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': accuracy,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Modality-Specific

```python
# Audio only
audio_outputs = model.forward_audio_only(audio)

# Visual only
visual_outputs = model.forward_visual_only(video)

# Multimodal
multi_outputs = model(audio, video)
```

## Memory Optimization

```python
# Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(audio, video)
    loss = criterion(outputs["logits"], labels)

# Gradient accumulation
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()

if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
# Apply to model layers
```

## Common Issues

### Out of Memory
```python
# Reduce batch size
batch_size = 2

# Use gradient accumulation
accumulation_steps = 4

# Enable mixed precision
use_amp = True

# Freeze some components
for param in model.audio_branch.parameters():
    param.requires_grad = False
```

### Slow Training
```python
# Use pretrained branches
audio_branch.load_state_dict(torch.load("audio_pretrained.pth"))
visual_branch.load_state_dict(torch.load("visual_pretrained.pth"))

# Freeze branches
for param in model.audio_branch.parameters():
    param.requires_grad = False
for param in model.visual_branch.parameters():
    param.requires_grad = False

# Train only fusion + classifier
```

### Poor Performance
```python
# Check data quality
# - Audio: 16kHz, 3 seconds
# - Video: 16 frames, 224x224

# Use label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Add data augmentation
# - Audio: time stretch, pitch shift
# - Video: random crop, flip

# Try different learning rates
lr_finder = ...  # Use learning rate finder
```

## File Structure

```
models/
├── audio_branch/          # Audio processing
├── visual_branch/         # Visual processing
├── fusion/                # LFM2 fusion
├── classifier.py          # Emotion classifier
└── multimodal_fer.py      # Complete model

configs/
├── data_config.yaml       # Data settings
├── model_config.yaml      # Model settings
└── train_config.yaml      # Training settings

scripts/
├── demo_complete_model.py # Demo
└── train.py               # Training (TODO)

tests/
└── test_complete_model.py # Tests
```

## Documentation

- `README.md` - Project overview
- `QUICK_START.md` - Getting started
- `TRAINING_GUIDE.md` - Training instructions
- `FUSION_IMPLEMENTATION_SUMMARY.md` - Fusion details
- `PROJECT_STATUS.md` - Progress tracking
- `HOAN_THANH_FUSION.md` - Vietnamese summary

## Commands

```bash
# Test model
python tests/test_complete_model.py

# Run demo
python scripts/demo_complete_model.py

# Train (TODO)
python scripts/train.py --config configs/train_config.yaml

# Evaluate (TODO)
python scripts/evaluate.py --checkpoint best_model.pth
```

## Emotion Labels (RAVDESS)

| Index | Emotion | Code |
|-------|---------|------|
| 0 | Neutral | 01 |
| 1 | Calm | 02 |
| 2 | Happy | 03 |
| 3 | Sad | 04 |
| 4 | Angry | 05 |
| 5 | Fearful | 06 |
| 6 | Disgust | 07 |
| 7 | Surprised | 08 |

## Expected Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Audio Only | 65-70% | 0.63 |
| Visual Only | 70-75% | 0.68 |
| **LFM2 Fusion** | **80-85%** | **0.78** |

## Resources

- LFM2: https://huggingface.co/LiquidAI/LFM2-700M
- SigLIP: https://huggingface.co/google/siglip-base-patch16-224
- RAVDESS: https://zenodo.org/record/1188976
