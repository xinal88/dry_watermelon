# LFM2 Fusion Module

## Overview

Multimodal fusion module using **Liquid LFM2-700M** architecture for combining audio and visual features in emotion recognition.

## Architecture

```
Audio Features [B, 8, 512] ──┐
                              │
                              ├──> Gated Projection ──> [B, 8, 1536]
                              │
Visual Features [B, 8, 768] ──┘
                              │
                              ├──> Add Modality Type Embeddings
                              │
                              ├──> Interleave [B, 16, 1536]
                              │
                              ├──> LFM2 Layers (6 layers):
                              │    ├─ Lfm2ShortConv (local patterns)
                              │    ├─ Lfm2Attention (global dependencies)
                              │    └─ Lfm2MLP (transformations)
                              │
                              ├──> Separate & Average [B, 8, 1536]
                              │
                              └──> Output Projection ──> [B, 8, 512]
```

## Components

### 1. **Modality Projection** (`ModalityProjection`)

Projects modality-specific features to LFM2 hidden dimension using gated mechanism:

```python
gate = sigmoid(Linear(x))
value = GELU(Linear(x))
output = Linear(gate * value)
```

### 2. **LFM2 Backbone**

Two options:

#### Option A: Pretrained LFM2-700M
```python
fusion = LFM2Fusion(
    pretrained_model="LiquidAI/LFM2-700M",
    use_pretrained=True,
    freeze_backbone=False,
    num_layers=6,
)
```

#### Option B: Custom LFM2 Layers
```python
fusion = LFM2Fusion(
    use_pretrained=False,
    hidden_dim=1536,
    num_layers=6,
)
```

### 3. **LFM2 Layers** (`lfm2_layers.py`)

#### Lfm2ShortConv
- Gated depthwise convolution
- Kernel size: 3
- Causal padding
- SiLU activation

#### Lfm2Attention
- Grouped query attention (GQA)
- 24 query heads, 8 key-value heads
- RMS normalization on Q and K
- Efficient for long sequences

#### Lfm2MLP
- SwiGLU activation: `SiLU(W1(x)) * W3(x)`
- Expansion factor: 4.5x
- No bias

## Usage

### Basic Usage

```python
from models.fusion import LFM2Fusion

# Create fusion module
fusion = LFM2Fusion(
    audio_dim=512,
    visual_dim=768,
    hidden_dim=1536,
    num_layers=6,
    output_dim=512,
)

# Forward pass
audio_features = torch.randn(4, 8, 512)
visual_features = torch.randn(4, 8, 768)

output = fusion(audio_features, visual_features)

# Output
fused_features = output["fused_features"]  # [4, 8, 512]
pooled_features = output["pooled_features"]  # [4, 512]
```

### With Configuration

```python
from models.fusion import LFM2Fusion, LFM2FusionConfig

config = LFM2FusionConfig(
    audio_dim=512,
    visual_dim=768,
    pretrained_model="LiquidAI/LFM2-700M",
    use_pretrained=True,
    freeze_backbone=False,
    num_layers=6,
    output_dim=512,
)

fusion = LFM2Fusion.from_config(config)
```

### Pretrained LFM2

```python
# Load pretrained LFM2-700M
fusion = LFM2Fusion(
    pretrained_model="LiquidAI/LFM2-700M",
    use_pretrained=True,
    freeze_backbone=True,  # Freeze for faster training
    num_layers=6,  # Use first 6 layers
)

# Check parameters
params = fusion.count_parameters()
print(f"Total: {params['total']:,} params")
print(f"Trainable: {params['total'] - params['backbone']:,} params")
```

## Configuration

### LFM2FusionConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_dim` | 512 | Audio feature dimension |
| `visual_dim` | 768 | Visual feature dimension |
| `num_segments` | 8 | Number of temporal segments |
| `pretrained_model` | "LiquidAI/LFM2-700M" | Pretrained model name |
| `use_pretrained` | True | Use pretrained weights |
| `freeze_backbone` | False | Freeze LFM2 layers |
| `hidden_dim` | 1536 | LFM2 hidden dimension |
| `num_layers` | 6 | Number of LFM2 layers |
| `dropout` | 0.1 | Dropout rate |
| `use_gated_projection` | True | Use gated projection |
| `projection_hidden_dim` | 1024 | Projection hidden dim |
| `output_dim` | 512 | Output feature dimension |

## Model Size

### Pretrained LFM2-700M (6 layers)
- **Backbone**: ~100M params
- **Projections**: ~3M params
- **Total**: ~103M params

### Custom LFM2 (6 layers)
- **Backbone**: ~15M params
- **Projections**: ~3M params
- **Total**: ~18M params

## Memory Usage

| Configuration | FP32 | FP16 |
|---------------|------|------|
| Pretrained (6 layers) | ~400 MB | ~200 MB |
| Custom (6 layers) | ~70 MB | ~35 MB |

## Training Tips

### 1. Pretrained Backbone

```python
# Start with frozen backbone
fusion = LFM2Fusion(
    use_pretrained=True,
    freeze_backbone=True,
)

# Train projections first
# ...

# Then unfreeze and finetune
for param in fusion.backbone.parameters():
    param.requires_grad = True
```

### 2. Differential Learning Rates

```python
param_groups = [
    {"params": fusion.audio_proj.parameters(), "lr": 1e-4},
    {"params": fusion.visual_proj.parameters(), "lr": 1e-4},
    {"params": fusion.backbone.parameters(), "lr": 1e-5},  # Lower LR
    {"params": fusion.output_proj.parameters(), "lr": 1e-4},
]

optimizer = torch.optim.AdamW(param_groups)
```

### 3. Gradient Checkpointing

```python
# For memory efficiency
from torch.utils.checkpoint import checkpoint

for layer in fusion.backbone:
    layer.forward = checkpoint(layer.forward)
```

## Performance

### RAVDESS Dataset (Expected)

| Configuration | Accuracy | F1-Score |
|---------------|----------|----------|
| Custom (4 layers) | ~78% | 0.75 |
| Custom (6 layers) | ~80% | 0.77 |
| Pretrained (6 layers) | ~82% | 0.79 |
| Pretrained (12 layers) | ~85% | 0.82 |

## References

- [Liquid Foundation Models](https://www.liquid.ai/)
- [LFM2-700M on HuggingFace](https://huggingface.co/LiquidAI/LFM2-700M)
- [LFM2 Technical Report](../../refs/paper/LFM2%20Technical%20Report.pdf)

## Examples

See:
- `scripts/demo_complete_model.py` - Complete model demo
- `tests/test_complete_model.py` - Unit tests
- `TRAINING_GUIDE.md` - Training instructions
