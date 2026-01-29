# Sơ Đồ Chi Tiết Kiến Trúc Mô Hình

## 1. Tổng Quan Luồng Dữ Liệu

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTIMODAL FER MODEL                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────────────┐                    ┌──────────────────────┐
│   Audio Waveform     │                    │    Video Frames      │
│   [B, T_audio]       │                    │  [B, T, 3, 224, 224] │
│   16kHz, ~3 seconds  │                    │   30fps, ~3 seconds  │
└──────────┬───────────┘                    └──────────┬───────────┘
           │                                           │
           ▼                                           ▼
┌──────────────────────┐                    ┌──────────────────────┐
│   AUDIO BRANCH       │                    │   VISUAL BRANCH      │
│   FastConformer      │                    │   SigLIP + Temporal  │
└──────────┬───────────┘                    └──────────┬───────────┘
           │                                           │
           │  [B, 8, 512]                             │  [B, 8, 768]
           │                                           │
           └───────────────┬───────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   LFM2 FUSION        │
                │   Liquid Neural Net  │
                └──────────┬───────────┘
                           │
                           │  [B, 8, 512]
                           │
                           ▼
                ┌──────────────────────┐
                │   CLASSIFIER         │
                │   MLP + Softmax      │
                └──────────┬───────────┘
                           │
                           ▼
OUTPUT:
┌──────────────────────────────────────────┐
│   Emotion Probabilities [B, 8]           │
│   [neutral, calm, happy, sad, angry,     │
│    fearful, disgust, surprised]          │
└──────────────────────────────────────────┘
```

---

## 2. Audio Branch Chi Tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO BRANCH                              │
└─────────────────────────────────────────────────────────────────┘

INPUT: Raw Audio [B, T_audio] (e.g., [4, 48000] for 3 seconds at 16kHz)
   │
   ▼
┌─────────────────────────────────────────┐
│  Audio Preprocessing                    │
│  ├─ Mel Spectrogram                     │
│  │  ├─ n_fft: 512                       │
│  │  ├─ hop_length: 160                  │
│  │  ├─ n_mels: 80                       │
│  │  └─ Output: [B, T_frames, 80]        │
│  └─ Normalization                       │
└─────────────┬───────────────────────────┘
              │  [B, T_frames, 80]  (e.g., [4, 300, 80])
              ▼
┌─────────────────────────────────────────┐
│  FastConformer Encoder (17 layers)      │
│  ┌─────────────────────────────────┐   │
│  │  Layer 1-17: Conformer Block    │   │
│  │  ├─ Multi-Head Self-Attention   │   │
│  │  │  └─ 8 heads, d_model=512     │   │
│  │  ├─ Convolution Module          │   │
│  │  │  └─ Depthwise Conv, k=31     │   │
│  │  ├─ Feed-Forward Module         │   │
│  │  │  └─ 4x expansion              │   │
│  │  └─ Layer Normalization         │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, T_frames, 512]
              ▼
┌─────────────────────────────────────────┐
│  Segment Attention Pooling              │
│  ┌─────────────────────────────────┐   │
│  │  1. Divide into 8 segments      │   │
│  │     T_frames → 8 segments       │   │
│  │                                  │   │
│  │  2. For each segment:           │   │
│  │     ├─ Multi-Head Attention     │   │
│  │     │  └─ Query: learnable      │   │
│  │     │  └─ Key/Value: frames     │   │
│  │     └─ Output: [B, 512]         │   │
│  │                                  │   │
│  │  3. Add temporal encoding       │   │
│  │     └─ Positional encoding      │   │
│  │        for segment order        │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │
              ▼
OUTPUT: Audio Features [B, 8, 512]
        8 segment-level representations
```

---

## 3. Visual Branch Chi Tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                        VISUAL BRANCH                             │
└─────────────────────────────────────────────────────────────────┘

INPUT: Video Frames [B, T, 3, 224, 224] (e.g., [4, 90, 3, 224, 224])
   │
   ▼
┌─────────────────────────────────────────┐
│  SigLIP Vision Encoder                  │
│  (google/siglip2-base-patch16-224)      │
│  ┌─────────────────────────────────┐   │
│  │  For each frame:                │   │
│  │  1. Patch Embedding             │   │
│  │     224x224 → 14x14 patches     │   │
│  │     Each patch: 16x16 pixels    │   │
│  │     Total: 196 patches          │   │
│  │                                  │   │
│  │  2. Vision Transformer          │   │
│  │     ├─ 12 layers                │   │
│  │     ├─ 12 heads                 │   │
│  │     ├─ d_model = 768            │   │
│  │     └─ Pretrained weights       │   │
│  │                                  │   │
│  │  3. Output: Patch Tokens        │   │
│  │     [196, 768] per frame        │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, T, 196, 768]
              ▼
┌─────────────────────────────────────────┐
│  ROI Token Compression                  │
│  ┌─────────────────────────────────┐   │
│  │  1. Compute Importance Scores   │   │
│  │     score = Linear(token)       │   │
│  │     If ROI mask provided:       │   │
│  │       score *= roi_weight       │   │
│  │                                  │   │
│  │  2. Select Top-K Tokens         │   │
│  │     ├─ Top 64 by score          │   │
│  │     └─ 4 global tokens          │   │
│  │        (mean, max, min, std)    │   │
│  │                                  │   │
│  │  3. Output: Compressed Tokens   │   │
│  │     196 → 68 tokens (65% ↓)     │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, T, 68, 768]
              ▼
┌─────────────────────────────────────────┐
│  Temporal Encoder (6 layers)            │
│  ┌─────────────────────────────────┐   │
│  │  Hybrid Architecture:           │   │
│  │                                  │   │
│  │  Layer 1-4: GSCB (70%)          │   │
│  │  ├─ Gated Convolution           │   │
│  │  │  └─ Kernel size: 4           │   │
│  │  ├─ Depthwise Conv1D            │   │
│  │  └─ SiLU activation             │   │
│  │                                  │   │
│  │  Layer 5-6: Attention (30%)     │   │
│  │  ├─ Multi-Head Attention        │   │
│  │  │  └─ 8 heads                  │   │
│  │  └─ Feed-Forward                │   │
│  │                                  │   │
│  │  All layers:                    │   │
│  │  └─ Residual connections        │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, T, 768]
              ▼
┌─────────────────────────────────────────┐
│  Segment Pooling                        │
│  ├─ Divide T frames into 8 segments     │
│  ├─ Average pooling per segment         │
│  └─ Output: [B, 8, 768]                 │
└─────────────┬───────────────────────────┘
              │
              ▼
OUTPUT: Visual Features [B, 8, 768]
        8 segment-level representations
```

---

## 4. LFM2 Fusion Chi Tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                        LFM2 FUSION                               │
└─────────────────────────────────────────────────────────────────┘

INPUT:
Audio Features [B, 8, 512]    Visual Features [B, 8, 768]
        │                              │
        ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ Gated Projection │          │ Gated Projection │
│ 512 → 1536       │          │ 768 → 1536       │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         │  [B, 8, 1536]                │  [B, 8, 1536]
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ + audio_type_    │          │ + visual_type_   │
│   embed          │          │   embed          │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Interleave Modalities       │
         │  [a1,v1,a2,v2,...,a8,v8]     │
         │  Shape: [B, 16, 1536]        │
         └──────────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────┐
│  LFM2 Backbone (6 layers)               │
│  ┌─────────────────────────────────┐   │
│  │  Layer 1: Lfm2ShortConv         │   │
│  │  ├─ Depthwise Conv1D (k=3)      │   │
│  │  ├─ Gating: SiLU(Linear(x))     │   │
│  │  └─ Residual connection         │   │
│  │                                  │   │
│  │  Layer 2: Lfm2ShortConv         │   │
│  │  └─ Same as Layer 1             │   │
│  │                                  │   │
│  │  Layer 3: Lfm2Attention         │   │
│  │  ├─ Grouped Query Attention     │   │
│  │  │  ├─ 24 query heads           │   │
│  │  │  └─ 8 key-value heads        │   │
│  │  ├─ RMS Norm on Q and K         │   │
│  │  └─ Residual connection         │   │
│  │                                  │   │
│  │  Layer 4-6: Repeat pattern      │   │
│  │  └─ Conv → Conv → Attention     │   │
│  │                                  │   │
│  │  Each layer also has:           │   │
│  │  └─ Lfm2MLP (SwiGLU)            │   │
│  │     └─ SiLU(W1(x)) * W3(x)      │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, 16, 1536]
              ▼
┌─────────────────────────────────────────┐
│  Separate & Combine                     │
│  ├─ Reshape: [B, 8, 2, 1536]            │
│  ├─ audio_fused = [:, :, 0, :]          │
│  ├─ visual_fused = [:, :, 1, :]         │
│  └─ combined = (audio + visual) / 2     │
└─────────────┬───────────────────────────┘
              │  [B, 8, 1536]
              ▼
┌─────────────────────────────────────────┐
│  Output Projection                      │
│  ├─ LayerNorm                           │
│  ├─ Linear(1536 → 512)                  │
│  ├─ GELU                                │
│  └─ Dropout                             │
└─────────────┬───────────────────────────┘
              │
              ▼
OUTPUT: Fused Features [B, 8, 512]
        Cross-modal representations
```

---

## 5. Classifier Chi Tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                        EMOTION CLASSIFIER                        │
└─────────────────────────────────────────────────────────────────┘

INPUT: Fused Features [B, 8, 512]
   │
   ▼
┌─────────────────────────────────────────┐
│  Temporal Pooling (Attention)           │
│  ┌─────────────────────────────────┐   │
│  │  Query: learnable [1, 512]      │   │
│  │  Key/Value: 8 segments          │   │
│  │                                  │   │
│  │  Attention(Q, K, V):            │   │
│  │  ├─ scores = Q @ K^T / √d       │   │
│  │  ├─ weights = softmax(scores)   │   │
│  │  └─ output = weights @ V        │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, 512]
              ▼
┌─────────────────────────────────────────┐
│  MLP Classifier                         │
│  ┌─────────────────────────────────┐   │
│  │  Layer 1:                       │   │
│  │  ├─ Linear(512 → 512)           │   │
│  │  ├─ LayerNorm(512)              │   │
│  │  ├─ GELU()                      │   │
│  │  └─ Dropout(0.1)                │   │
│  │                                  │   │
│  │  Layer 2:                       │   │
│  │  ├─ Linear(512 → 256)           │   │
│  │  ├─ LayerNorm(256)              │   │
│  │  ├─ GELU()                      │   │
│  │  └─ Dropout(0.1)                │   │
│  │                                  │   │
│  │  Output Layer:                  │   │
│  │  └─ Linear(256 → 8)             │   │
│  └─────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │  [B, 8] (logits)
              ▼
┌─────────────────────────────────────────┐
│  Softmax                                │
│  └─ probabilities = softmax(logits)     │
└─────────────┬───────────────────────────┘
              │
              ▼
OUTPUT: 
┌─────────────────────────────────────────┐
│  Emotion Probabilities [B, 8]           │
│  ┌─────────────────────────────────┐   │
│  │  0: neutral                     │   │
│  │  1: calm                        │   │
│  │  2: happy                       │   │
│  │  3: sad                         │   │
│  │  4: angry                       │   │
│  │  5: fearful                     │   │
│  │  6: disgust                     │   │
│  │  7: surprised                   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 6. Tensor Shape Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE SHAPE FLOW                            │
└──────────────────────────────────────────────────────────────────┘

Audio Path:
[B, 48000]              Raw audio (3s @ 16kHz)
    ↓
[B, 300, 80]            Mel spectrogram
    ↓
[B, 300, 512]           FastConformer output
    ↓
[B, 8, 512]             Segment pooling
    ↓
[B, 8, 1536]            Gated projection
    ↓
                        ┌─────────────┐
                        │   Fusion    │
                        └─────────────┘
                               ↓
Visual Path:                   ↓
[B, 90, 3, 224, 224]    Video frames (3s @ 30fps)
    ↓
[B, 90, 196, 768]       SigLIP patch tokens
    ↓
[B, 90, 68, 768]        ROI compression
    ↓
[B, 90, 768]            Temporal encoder
    ↓
[B, 8, 768]             Segment pooling
    ↓
[B, 8, 1536]            Gated projection
    ↓
                        ┌─────────────┐
                        │   Fusion    │
                        └─────────────┘
                               ↓
                        [B, 16, 1536]  Interleaved
                               ↓
                        [B, 16, 1536]  LFM2 layers
                               ↓
                        [B, 8, 1536]   Separate & combine
                               ↓
                        [B, 8, 512]    Output projection
                               ↓
                        [B, 512]       Temporal pooling
                               ↓
                        [B, 8]         Classifier output
```

---

## 7. Parameter Distribution

```
┌──────────────────────────────────────────────────────────────────┐
│                    PARAMETER BREAKDOWN                            │
└──────────────────────────────────────────────────────────────────┘

Total Model: ~243M parameters (with pretrained LFM2)

┌─────────────────────────────────────────────────────────┐
│  Audio Branch: ~50M (20.6%)                             │
│  ├─ FastConformer: 48M                                  │
│  └─ Segment Pooling: 2M                                 │
├─────────────────────────────────────────────────────────┤
│  Visual Branch: ~90M (37.0%)                            │
│  ├─ SigLIP Encoder: 86M                                 │
│  ├─ ROI Compression: 1M                                 │
│  └─ Temporal Encoder: 3M                                │
├─────────────────────────────────────────────────────────┤
│  LFM2 Fusion: ~103M (42.4%)                             │
│  ├─ Audio Projection: 1.5M                              │
│  ├─ Visual Projection: 1.5M                             │
│  ├─ LFM2 Backbone: 100M                                 │
│  └─ Output Projection: 0.8M                             │
├─────────────────────────────────────────────────────────┤
│  Classifier: ~0.5M (0.2%)                               │
│  ├─ MLP Layers: 0.4M                                    │
│  └─ Temporal Pooling: 0.1M                              │
└─────────────────────────────────────────────────────────┘

Visual Representation:
Audio Branch    ████████████████████ 20.6%
Visual Branch   █████████████████████████████████████ 37.0%
LFM2 Fusion     ██████████████████████████████████████████ 42.4%
Classifier      █ 0.2%
```

---

## 8. Computational Cost

```
┌──────────────────────────────────────────────────────────────────┐
│                    FLOPS BREAKDOWN (per sample)                   │
└──────────────────────────────────────────────────────────────────┘

Audio Branch:
├─ Mel Spectrogram:      ~0.1 GFLOPs
├─ FastConformer:        ~15 GFLOPs
└─ Segment Pooling:      ~0.5 GFLOPs
Total:                   ~15.6 GFLOPs

Visual Branch:
├─ SigLIP (90 frames):   ~180 GFLOPs  (2 GFLOPs × 90)
├─ ROI Compression:      ~1 GFLOPs
└─ Temporal Encoder:     ~5 GFLOPs
Total:                   ~186 GFLOPs

LFM2 Fusion:
├─ Projections:          ~0.5 GFLOPs
├─ LFM2 Layers:          ~8 GFLOPs
└─ Output Projection:    ~0.2 GFLOPs
Total:                   ~8.7 GFLOPs

Classifier:
└─ MLP:                  ~0.3 GFLOPs

TOTAL:                   ~210 GFLOPs per sample

Inference Time (RTX 3050):
├─ Batch size 1:         ~150ms
├─ Batch size 4:         ~400ms
└─ Batch size 8:         ~700ms
```

---

## 9. Memory Usage

```
┌──────────────────────────────────────────────────────────────────┐
│                    MEMORY BREAKDOWN (FP16)                        │
└──────────────────────────────────────────────────────────────────┘

Model Weights:           ~480 MB
├─ Audio Branch:         ~100 MB
├─ Visual Branch:        ~180 MB
├─ LFM2 Fusion:          ~200 MB
└─ Classifier:           ~1 MB

Activations (batch=4):   ~2 GB
├─ Audio features:       ~50 MB
├─ Visual features:      ~1.5 GB
├─ Fusion features:      ~100 MB
└─ Gradients:            ~480 MB

Total Training (batch=4): ~3 GB
Total Inference (batch=4): ~2.5 GB

Recommended GPU:
├─ Training:             8GB+ VRAM (RTX 3060 or better)
└─ Inference:            4GB+ VRAM (RTX 3050 or better)
```

Đây là sơ đồ chi tiết và dễ hiểu về kiến trúc mô hình của bạn!
