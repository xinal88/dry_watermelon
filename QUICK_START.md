# Quick Start Guide - Audio Branch

## 🚀 Installation

```bash
# Clone repository
cd f:\AI_project\dry_watermelon

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytest matplotlib einops

# Optional: For NVIDIA NeMo support
# pip install nemo_toolkit[asr]
```

## ✅ Verify Installation

```bash
# Run tests
python tests/test_audio_branch.py

# Run demo
python scripts/demo_audio_branch.py
```

## 📝 Basic Usage

### 1. Simple Example

```python
import torch
from models.audio_branch import AudioBranch

# Create audio branch
audio_branch = AudioBranch(
    feature_dim=512,
    num_segments=8,
    num_layers=4,  # Lightweight for RTX 3050
    pooling_type="attention"
)

# Process audio (3 seconds at 16kHz)
audio = torch.randn(4, 48000)  # [batch_size, audio_length]
output = audio_branch(audio)

print(output["segment_features"].shape)  # [4, 8, 512]
```

### 2. With Configuration

```python
from models.audio_branch import AudioBranch, AudioBranchConfig

# Create configuration
config = AudioBranchConfig(
    feature_dim=512,
    num_segments=8,
    num_layers=4,
    d_model=512,
    num_heads=8,
    pooling_type="attention",
    sample_rate=16000,
    n_mels=80
)

# Build model
audio_branch = AudioBranch(**config.to_dict())

# Process audio
audio = torch.randn(2, 48000)
output = audio_branch(audio, return_all_segments=True)
```

### 3. Load Real Audio File

```python
import torchaudio
from models.audio_branch import AudioBranch

# Load audio
waveform, sample_rate = torchaudio.load("path/to/audio.wav")

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Ensure mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Add batch dimension
waveform = waveform.unsqueeze(0)  # [1, T]

# Process
audio_branch = AudioBranch(feature_dim=512, num_segments=8)
output = audio_branch(waveform)
```

## 🎯 Common Use Cases

### For Development (RTX 3050)
```python
config = AudioBranchConfig(
    num_layers=4,        # Lightweight
    feature_dim=512,
    num_segments=8,
    pooling_type="attention"
)
```

### For Production (High Performance)
```python
config = AudioBranchConfig(
    pretrained_model="nvidia/stt_en_fastconformer_ctc_large",
    freeze_encoder=True,  # Freeze pretrained weights
    use_nemo=True,
    num_layers=17,
    feature_dim=512,
    num_segments=8
)
```

### For Fast Inference
```python
config = AudioBranchConfig(
    num_layers=4,
    feature_dim=256,      # Smaller dimension
    num_segments=4,       # Fewer segments
    pooling_type="max"    # Faster than attention
)
```

## 📊 Output Format

```python
output = audio_branch(audio, return_all_segments=True)

# Output dictionary contains:
{
    "segment_features": torch.Tensor,  # [B, S, D] - Main output
    "encoder_features": torch.Tensor,  # [B, T, D] - Raw encoder output
    "num_segments": int                # Number of segments (S)
}

# For single representation:
output = audio_branch(audio, return_all_segments=False)
# output["segment_features"]: [B, D] - Mean pooled across segments
```

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dim` | 512 | Output feature dimension |
| `num_segments` | 8 | Number of temporal segments |
| `num_layers` | 17 | Number of Conformer blocks |
| `d_model` | 512 | Model hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `pooling_type` | "attention" | Pooling strategy |
| `sample_rate` | 16000 | Audio sample rate (Hz) |
| `n_mels` | 80 | Number of mel filterbanks |

## 🧪 Testing

```bash
# Run all tests
python tests/test_audio_branch.py

# Run specific test
pytest tests/test_audio_branch.py::TestAudioBranch::test_audio_branch_forward

# Run with verbose output
pytest tests/test_audio_branch.py -v
```

## 📈 Performance Tips

1. **Reduce layers for development**: Use `num_layers=4-8` instead of 17
2. **Use smaller batch sizes**: Start with `batch_size=4` on RTX 3050
3. **Freeze pretrained encoder**: Set `freeze_encoder=True` to save memory
4. **Use max pooling for speed**: Set `pooling_type="max"` for faster inference
5. **Reduce segments**: Use `num_segments=4` instead of 8 to save memory

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Solution 1: Reduce batch size
# Solution 2: Reduce num_layers
# Solution 3: Use gradient checkpointing
audio_branch = AudioBranch(num_layers=4, feature_dim=256)
```

### Slow inference
```python
# Use max pooling instead of attention
audio_branch = AudioBranch(pooling_type="max")

# Reduce number of segments
audio_branch = AudioBranch(num_segments=4)
```

### Import errors
```bash
# Make sure you're in the project root
cd f:\AI_project\dry_watermelon

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"  # Windows PowerShell
```

## 📚 Next Steps

1. ✅ Audio Branch implemented
2. ⏳ Implement Visual Branch (SigLip2 + ROI + Temporal Encoder)
3. ⏳ Implement Liquid Fusion
4. ⏳ Create RAVDESS dataset loader
5. ⏳ Build training pipeline

## 💡 Tips

- Start with the demo script to understand the pipeline
- Check `models/audio_branch/README.md` for detailed documentation
- Use configuration files in `configs/` for reproducibility
- Monitor GPU memory with `nvidia-smi`

