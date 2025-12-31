# 🎭 Multimodal Facial Expression Recognition

Deep learning model for emotion recognition using audio and video modalities.

## 📊 Overview

- **Task**: Facial Expression Recognition (FER)
- **Dataset**: RAVDESS (1440 videos, 8 emotions)
- **Modalities**: Audio + Video
- **Architecture**: FastConformer + SigLIP2 + LFM2 Fusion
- **Performance**: 75-85% UAR

## 🏗️ Architecture

```
Audio Branch (FastConformer)
  └─ Segment Attention Pooling → [B, 8, 512]
                                      ↓
Video Branch (SigLIP2/Custom CNN)
  └─ ROI Compression + Temporal → [B, 8, 768]
                                      ↓
                    LFM2 Fusion (Liquid Neural Network)
                                      ↓
                    Classifier MLP → 8 Emotions
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-fer.git
cd multimodal-fer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test with Sample Data

```bash
# Test model
python scripts/quick_test.py

# Test inference
python scripts/inference_cpu.py
```

### 4. Train on Google Colab

See [GITHUB_COLAB_SETUP.md](GITHUB_COLAB_SETUP.md) for detailed instructions.

## 📁 Project Structure

```
multimodal-fer/
├── models/                 # Model architectures
│   ├── audio_branch/      # Audio processing
│   ├── visual_branch/     # Video processing
│   ├── fusion/            # LFM2 fusion
│   └── classifier.py      # Emotion classifier
├── training/              # Training utilities
│   ├── losses.py         # Loss functions
│   └── metrics.py        # Evaluation metrics
├── data/                  # Dataset loaders
│   ├── ravdess_dataset.py
│   └── test_dataset.py
├── scripts/               # Training & inference scripts
│   ├── train_cpu.py
│   ├── inference_cpu.py
│   └── evaluate.py
└── configs/               # Configuration files
```

## 🎯 Training

### On Google Colab (Recommended)

1. Upload RAVDESS data to Google Drive
2. Open `Train_Multimodal_FER.ipynb` in Colab
3. Select GPU runtime (T4 or A100)
4. Run all cells

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for details.

### On Local Machine

```bash
# Lightweight version (CPU)
python scripts/train_cpu.py

# GPU version (requires CUDA)
python scripts/train_lightweight.py
```

## 📊 Results

| Model | UAR | WAR | WA-F1 | Params |
|-------|-----|-----|-------|--------|
| Lightweight | 75-80% | 75-80% | 73-78% | ~150M |
| Full Pretrained | 80-85% | 80-85% | 78-83% | ~400M |

## 🔍 Inference

```python
from scripts.inference_cpu import EmotionPredictor

# Load model
predictor = EmotionPredictor(CONFIG)

# Predict
result = predictor.predict("path/to/video.mp4")

# Output:
# {
#   "predicted_emotion": "happy",
#   "confidence": 0.95,
#   "top_k": [...]
# }
```

## 📚 Documentation

- [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) - Train on Colab
- [GITHUB_COLAB_SETUP.md](GITHUB_COLAB_SETUP.md) - GitHub + Colab workflow
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training details
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Inference usage

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- FFmpeg (for audio extraction)

See [requirements.txt](requirements.txt) for full list.

## 📝 Citation

```bibtex
@misc{multimodal-fer-2024,
  title={Multimodal Facial Expression Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/multimodal-fer}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- RAVDESS Dataset
- FastConformer (NVIDIA NeMo)
- SigLIP2 (Google Research)
- LFM2 (Liquid AI)

## 📞 Contact

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

**⭐ Star this repo if you find it useful!**
