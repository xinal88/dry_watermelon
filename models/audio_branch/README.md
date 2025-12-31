# Audio Branch - FastConformer + Segment Attention Pooling

## 🎯 Overview

The Audio Branch processes audio signals for emotion recognition using a two-stage architecture:

1. **FastConformer Encoder**: Extracts temporal audio features
2. **Segment Attention Pooling**: Aggregates features into fixed-length segment representations

## 🏗️ Architecture

```
Raw Audio [B, T_audio]
    ↓
Mel Spectrogram [B, T, 80]
    ↓
FastConformer Encoder [B, T, 512]
    ↓
Segment Attention Pooling [B, 8, 512]
    ↓
Audio Features
```

### Components

#### 1. FastConformer Encoder
- **Backend Options**:
  - NVIDIA NeMo (recommended for production)
  - HuggingFace Transformers (fallback)
  - Custom lightweight implementation (for development)
  
- **Features**:
  - Efficient Conformer architecture
  - Combines convolution and self-attention
  - Optimized for speech/audio processing

#### 2. Segment Attention Pooling
- **Process**: `[B, T, D] → Split into S segments → Attention pool per segment → [B, S, D]`
- **Pooling Types**:
  - `attention`: Learnable attention-based pooling (default)
  - `max`: Max pooling per segment
  - `avg`: Average pooling per segment
  - `learnable`: Learnable weighted pooling

## 📦 Usage

### Basic Usage

```python
from models.audio_branch import AudioBranch

# Create audio branch
audio_branch = AudioBranch(
    pretrained_model=None,  # or "nvidia/stt_en_fastconformer_ctc_large"
    feature_dim=512,
    num_segments=8,
    pooling_type="attention",
    use_nemo=False  # Set True to use NVIDIA NeMo
)

# Process audio
import torch
audio = torch.randn(4, 48000)  # [batch_size, audio_length]

output = audio_branch(audio, return_all_segments=True)
# output["segment_features"]: [4, 8, 512]
```

### Using Configuration

```python
from models.audio_branch import AudioBranch, AudioBranchConfig

# Create config
config = AudioBranchConfig(
    feature_dim=512,
    num_segments=8,
    num_layers=17,
    d_model=512,
    num_heads=8,
    pooling_type="attention"
)

# Build model from config
audio_branch = AudioBranch(**config.to_dict())
```

### With Pretrained FastConformer (NeMo)

```python
audio_branch = AudioBranch(
    pretrained_model="nvidia/stt_en_fastconformer_ctc_large",
    feature_dim=512,
    freeze_encoder=True,  # Freeze pretrained weights
    use_nemo=True,
    num_segments=8
)
```

## 🧪 Testing

Run unit tests:
```bash
cd tests
python test_audio_branch.py
```

Run demo:
```bash
cd scripts
python demo_audio_branch.py
```

## 📊 Model Statistics

### Custom Implementation (4 layers)
- **Total Parameters**: ~8M
- **Encoder**: ~7.5M
- **Pooling**: ~0.5M

### Full Implementation (17 layers)
- **Total Parameters**: ~30M
- **Encoder**: ~28M
- **Pooling**: ~0.5M

## ⚙️ Configuration Parameters

### FastConformer
- `pretrained_model`: Path to pretrained model or None
- `feature_dim`: Output feature dimension (default: 512)
- `freeze_encoder`: Freeze encoder weights (default: False)
- `num_layers`: Number of Conformer blocks (default: 17)
- `d_model`: Model dimension (default: 512)
- `num_heads`: Number of attention heads (default: 8)

### Audio Preprocessing
- `sample_rate`: Audio sample rate (default: 16000)
- `n_mels`: Number of mel filterbanks (default: 80)
- `n_fft`: FFT size (default: 512)
- `hop_length`: Hop length for STFT (default: 160)
- `win_length`: Window length for STFT (default: 400)

### Segment Pooling
- `num_segments`: Number of segments to split sequence (default: 8)
- `pooling_type`: Type of pooling ("attention", "max", "avg", "learnable")
- `use_temporal_encoding`: Add positional encoding to segments (default: True)

## 🔧 Customization

### Custom Pooling Strategy

```python
from models.audio_branch import SegmentAttentionPooling

pooling = SegmentAttentionPooling(
    dim=512,
    num_segments=8,
    pooling_type="attention",
    num_heads=8,
    dropout=0.1
)
```

### Adjust Number of Segments

```python
# More segments = finer temporal resolution
audio_branch = AudioBranch(
    num_segments=16,  # Instead of 8
    feature_dim=512
)
```

## 📝 Notes

- For RTX 3050 (12GB), use `num_layers=4-8` for development
- For production, use pretrained NeMo models with `freeze_encoder=True`
- Segment pooling reduces sequence length from ~300 to 8, saving memory
- Attention pooling performs best but is slower than max/avg pooling

## 🚀 Next Steps

- [ ] Integrate with Visual Branch
- [ ] Add Liquid Fusion module
- [ ] Implement full training pipeline
- [ ] Add RAVDESS dataset loader

