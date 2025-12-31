# Lightweight Multimodal Dynamic Facial Expression Recognition

## 🎯 Project Overview
A lightweight multimodal deep learning model for Dynamic Facial Expression Recognition (DFER) combining:
- **Visual Branch**: SigLip2 + ROI-aware Token Compression + Temporal Encoder
- **Audio Branch**: FastConformer + Segment Attention Pooling
- **Fusion**: Liquid Neural Network
- **Target**: < 800M parameters, optimized for RTX 3050 (12GB)

## 📊 Datasets
- **Primary (Testing)**: RAVDESS
- **Extended**: CREMA-D, DFEW, MELD, AVFW

## 🏗️ Project Structure
```
dry_watermelon/
├── configs/                 # Configuration files
│   ├── model_config.yaml
│   ├── train_config.yaml
│   └── data_config.yaml
├── data/                    # Data processing
│   ├── datasets/
│   │   ├── ravdess.py
│   │   └── base_dataset.py
│   ├── preprocessing/
│   │   ├── audio_processor.py
│   │   └── video_processor.py
│   └── dataloader.py
├── models/                  # Model architectures
│   ├── audio_branch/
│   │   ├── fastconformer.py
│   │   └── segment_pooling.py
│   ├── visual_branch/
│   │   ├── siglip_encoder.py
│   │   ├── roi_compression.py
│   │   └── temporal_encoder.py
│   ├── fusion/
│   │   └── liquid_fusion.py
│   └── multimodal_fer.py
├── training/                # Training pipeline
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── utils/                   # Utilities
│   ├── logger.py
│   ├── checkpoint.py
│   └── visualization.py
├── scripts/                 # Execution scripts
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/                   # Unit tests
│   ├── test_audio_branch.py
│   └── test_models.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pth
```

## 📝 Development Progress
- [x] Project structure setup
- [x] Audio Branch implementation
- [ ] Visual Branch implementation
- [ ] Liquid Fusion implementation
- [ ] Training pipeline
- [ ] Evaluation pipeline

## 💻 Hardware Requirements
- **Development**: RTX 3050 (12GB VRAM)
- **Training (Extended)**: Google Colab Pro

## 📚 References
- FastConformer: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- SigLip2: [Google Research](https://arxiv.org/abs/2303.15343)
- Liquid Neural Networks: [MIT CSAIL](https://arxiv.org/abs/2006.04439)

