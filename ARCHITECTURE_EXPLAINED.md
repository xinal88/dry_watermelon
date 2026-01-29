# Giải Thích Kiến Trúc Mô Hình Multimodal FER

## Tổng Quan

Đây là một mô hình **Multimodal Facial Expression Recognition (FER)** kết hợp **Audio** và **Video** để nhận diện cảm xúc. Mô hình sử dụng kiến trúc tiên tiến với các thành phần chính:

```
Audio [B, T_audio] ──────> Audio Branch ──────> [B, 8, 512] ──┐
                                                                ├──> LFM2 Fusion ──> [B, 8, 512] ──> Classifier ──> [B, 8]
Video [B, T, 3, H, W] ──> Visual Branch ──────> [B, 8, 768] ──┘
```

**Kích thước:**
- B = Batch size
- T = Số frame video
- T_audio = Độ dài audio
- 8 = Số segments (chia video/audio thành 8 phần)
- 512/768 = Số chiều đặc trưng
- 8 = Số lớp cảm xúc (RAVDESS: neutral, calm, happy, sad, angry, fearful, disgust, surprised)

---

## 1. Audio Branch (Nhánh Âm Thanh)

### Pipeline:
```
Raw Audio [B, T_audio] 
  ↓
Mel Spectrogram [B, T, 80]
  ↓
FastConformer Encoder [B, T, 512]
  ↓
Segment Attention Pooling [B, 8, 512]
  ↓
Audio Features [B, 8, 512]
```

### Thành phần:

#### 1.1. FastConformer Encoder
- **Mục đích**: Trích xuất đặc trưng từ audio
- **Kiến trúc**: 
  - 17 layers Conformer (kết hợp CNN và Transformer)
  - d_model = 512
  - 8 attention heads
- **Input**: Mel spectrogram (80 mel bins)
- **Output**: Frame-level features [B, T, 512]

**Conformer = Convolution + Transformer:**
- Convolution: Bắt các pattern cục bộ trong audio
- Self-Attention: Bắt mối quan hệ toàn cục giữa các frame

#### 1.2. Segment Attention Pooling
- **Mục đích**: Chia audio thành 8 segments và tổng hợp thông tin
- **Cách hoạt động**:
  - Chia T frames thành 8 segments
  - Dùng attention để pooling mỗi segment
  - Tạo temporal encoding cho mỗi segment
- **Output**: [B, 8, 512] - 8 đặc trưng cho 8 phần của audio

---

## 2. Visual Branch (Nhánh Hình Ảnh)

### Pipeline:
```
Video Frames [B, T, 3, 224, 224]
  ↓
SigLIP Encoder [B, T, 196, 768]  (196 = 14x14 patches)
  ↓
ROI Token Compression [B, T, 68, 768]  (64 + 4 tokens)
  ↓
Temporal Encoder [B, T, 768]
  ↓
Segment Pooling [B, 8, 768]
  ↓
Visual Features [B, 8, 768]
```

### Thành phần:

#### 2.1. SigLIP Encoder
- **Mục đích**: Trích xuất đặc trưng từ mỗi frame
- **Model**: google/siglip2-base-patch16-224
- **Cách hoạt động**:
  - Chia mỗi frame 224x224 thành 14x14 = 196 patches (mỗi patch 16x16)
  - Mỗi patch được encode thành vector 768 chiều
- **Output**: [B, T, 196, 768] - 196 patch tokens cho mỗi frame

#### 2.2. ROI Token Compression
- **Mục đích**: Giảm số lượng tokens, tập trung vào vùng quan trọng (khuôn mặt)
- **Cách hoạt động**:
  - Tính importance score cho mỗi patch
  - Nếu có ROI mask (vùng khuôn mặt), tăng trọng số cho các patch trong ROI
  - Giữ lại 64 tokens quan trọng nhất + 4 global tokens
- **Output**: [B, T, 68, 768] - Giảm từ 196 xuống 68 tokens

**Tại sao cần compression?**
- Giảm computational cost
- Tập trung vào vùng khuôn mặt (quan trọng cho emotion recognition)
- 68 tokens đủ để giữ thông tin quan trọng

#### 2.3. Temporal Encoder
- **Mục đích**: Mô hình hóa động thái thời gian giữa các frames
- **Kiến trúc**: Hybrid GSCB + Attention
  - **GSCB (Gated Selective Convolution Block)**: 70% layers
    - Convolution 1D để bắt local temporal patterns
    - Gating mechanism để chọn lọc thông tin
  - **Multi-Head Attention**: 30% layers
    - Bắt long-range dependencies giữa các frames
- **Segment Pooling**: Chia T frames thành 8 segments
- **Output**: [B, 8, 768] - 8 đặc trưng cho 8 phần của video

---

## 3. LFM2 Fusion (Kết Hợp Đa Phương Thức)

### Pipeline:
```
Audio [B, 8, 512] ──> Gated Projection ──> [B, 8, 1536] ──┐
                                                            ├──> Add Type Embeddings
Visual [B, 8, 768] ──> Gated Projection ──> [B, 8, 1536] ──┘
                                                            ↓
                                            Interleave [B, 16, 1536]
                                                            ↓
                                            LFM2 Layers (6 layers)
                                                            ↓
                                            Separate & Average [B, 8, 1536]
                                                            ↓
                                            Output Projection [B, 8, 512]
```

### Thành phần:

#### 3.1. Modality Projection
- **Mục đích**: Chiếu audio và visual về cùng không gian 1536 chiều
- **Gated Projection**:
  ```python
  gate = sigmoid(Linear(x))
  value = GELU(Linear(x))
  output = Linear(gate * value)
  ```
- Gating giúp mô hình học được cách kết hợp thông tin

#### 3.2. Modality Type Embeddings
- Thêm learnable embeddings để phân biệt audio và visual
- Audio: thêm audio_type_embed
- Visual: thêm visual_type_embed

#### 3.3. LFM2 Backbone
**LFM2 = Liquid Foundation Model 2** (từ Liquid AI)

Có 2 options:
- **Option A**: Pretrained LFM2-700M (100M params cho 6 layers)
- **Option B**: Custom LFM2 (15M params cho 6 layers)

**Mỗi LFM2 Layer gồm:**

1. **Lfm2ShortConv** (Convolution layers):
   - Depthwise convolution với kernel size 3
   - Gating mechanism với SiLU activation
   - Bắt local patterns trong sequence

2. **Lfm2Attention** (Attention layers):
   - Grouped Query Attention (GQA)
   - 24 query heads, 8 key-value heads
   - RMS normalization
   - Bắt global dependencies

3. **Lfm2MLP** (Feed-forward):
   - SwiGLU activation: `SiLU(W1(x)) * W3(x)`
   - Expansion factor 4.5x
   - Transform features

**Tại sao dùng LFM2?**
- Liquid Neural Networks: Hiệu quả cho sequential data
- Kết hợp tốt convolution (local) và attention (global)
- Pretrained trên large-scale data

#### 3.4. Fusion Strategy
1. Interleave audio và visual: [a1, v1, a2, v2, ..., a8, v8]
2. Pass qua LFM2 layers để học cross-modal interactions
3. Separate lại thành audio và visual
4. Average fusion: `(audio_fused + visual_fused) / 2`

---

## 4. Emotion Classifier (Phân Loại Cảm Xúc)

### Pipeline:
```
Fused Features [B, 8, 512]
  ↓
Temporal Pooling [B, 512]
  ↓
MLP [512 -> 512 -> 256 -> 8]
  ↓
Logits [B, 8]
  ↓
Softmax
  ↓
Probabilities [B, 8]
```

### Thành phần:

#### 4.1. Temporal Pooling
Có 4 strategies:
- **Mean**: Average across 8 segments
- **Max**: Max pooling
- **Last**: Lấy segment cuối
- **Attention**: Learnable attention pooling (best)

#### 4.2. MLP Classifier
```
Input [512]
  ↓
Linear(512, 512) + LayerNorm + GELU + Dropout
  ↓
Linear(512, 256) + LayerNorm + GELU + Dropout
  ↓
Linear(256, 8)
  ↓
Output [8] (logits cho 8 emotions)
```

---

## 5. Kiến Trúc Tổng Thể

### Complete Forward Pass:

```python
# Input
audio = [B, T_audio]  # Raw waveform
video = [B, T, 3, 224, 224]  # Video frames

# 1. Audio Branch
audio_features = audio_branch(audio)  # [B, 8, 512]

# 2. Visual Branch  
visual_features = visual_branch(video)  # [B, 8, 768]

# 3. LFM2 Fusion
fused_features = fusion(audio_features, visual_features)  # [B, 8, 512]

# 4. Classifier
logits = classifier(fused_features)  # [B, 8]
probabilities = softmax(logits)  # [B, 8]
```

### Model Size:

| Component | Parameters | Memory (FP16) |
|-----------|-----------|---------------|
| Audio Branch | ~50M | ~100 MB |
| Visual Branch | ~90M | ~180 MB |
| LFM2 Fusion (pretrained) | ~103M | ~200 MB |
| LFM2 Fusion (custom) | ~18M | ~35 MB |
| Classifier | ~0.5M | ~1 MB |
| **Total (pretrained)** | **~243M** | **~480 MB** |
| **Total (custom)** | **~158M** | **~316 MB** |

---

## 6. Ưu Điểm Của Kiến Trúc

### 6.1. Multimodal Fusion
- Kết hợp audio và visual để có thông tin đầy đủ hơn
- Audio: giọng nói, ngữ điệu
- Visual: biểu cảm khuôn mặt, cử chỉ

### 6.2. Segment-based Processing
- Chia video/audio thành 8 segments
- Bắt được sự thay đổi cảm xúc theo thời gian
- Giảm computational cost so với frame-by-frame

### 6.3. ROI-aware Compression
- Tập trung vào vùng khuôn mặt
- Giảm 196 tokens xuống 68 tokens (65% reduction)
- Giữ được thông tin quan trọng

### 6.4. Hybrid Temporal Modeling
- GSCB: Efficient cho local patterns
- Attention: Powerful cho long-range dependencies
- Kết hợp tốt nhất của cả hai

### 6.5. LFM2 Fusion
- State-of-the-art liquid neural network
- Có thể dùng pretrained weights
- Hiệu quả cho multimodal fusion

---

## 7. Training Strategy

### 7.1. Data Flow
```
RAVDESS Dataset
  ├─ Audio: 16kHz WAV files
  └─ Video: 30fps MP4 files
       ↓
  Preprocessing
  ├─ Audio: Resample to 16kHz
  └─ Video: Extract frames, resize to 224x224
       ↓
  DataLoader
  ├─ Audio: [B, T_audio]
  └─ Video: [B, T, 3, 224, 224]
       ↓
  Model
       ↓
  Loss: CrossEntropyLoss
       ↓
  Optimizer: AdamW
```

### 7.2. Training Tips

1. **Pretrained Components**:
   - SigLIP: Pretrained on image-text pairs
   - FastConformer: Pretrained on speech
   - LFM2: Pretrained on large-scale sequences

2. **Freezing Strategy**:
   ```python
   # Stage 1: Freeze encoders, train fusion + classifier
   freeze_encoder = True
   
   # Stage 2: Unfreeze all, finetune end-to-end
   freeze_encoder = False
   ```

3. **Differential Learning Rates**:
   ```python
   param_groups = [
       {"params": audio_branch.parameters(), "lr": 1e-5},
       {"params": visual_branch.parameters(), "lr": 1e-5},
       {"params": fusion.parameters(), "lr": 1e-4},
       {"params": classifier.parameters(), "lr": 1e-4},
   ]
   ```

4. **Data Augmentation**:
   - Audio: Time stretching, pitch shifting, noise
   - Video: Random crop, color jitter, horizontal flip

---

## 8. Inference

### 8.1. Single Sample
```python
model.eval()
with torch.no_grad():
    output = model(audio, video)
    probs = output["probabilities"]
    pred = probs.argmax(dim=-1)
    emotion = model.get_emotion_labels()[pred]
```

### 8.2. Batch Inference
```python
for batch in dataloader:
    audio, video, labels = batch
    output = model(audio, video)
    # Process outputs
```

---

## 9. Tài Liệu Tham Khảo

### Papers:
- **FastConformer**: "Fast Conformer with Linearly Scalable Attention"
- **SigLIP**: "Sigmoid Loss for Language Image Pre-Training"
- **LFM2**: "Liquid Foundation Models 2" (Liquid AI)

### Code:
- `models/multimodal_fer.py`: Complete model
- `models/audio_branch/`: Audio processing
- `models/visual_branch/`: Visual processing
- `models/fusion/`: LFM2 fusion
- `models/classifier.py`: Emotion classifier

### Training:
- `scripts/train_simple.py`: Simple training script
- `scripts/train_ravdess.py`: Full RAVDESS training
- `TRAINING_GUIDE.md`: Detailed training guide

---

## 10. Kết Luận

Đây là một kiến trúc **state-of-the-art** cho multimodal emotion recognition với:

✅ **Hiệu quả**: Segment-based processing, ROI compression
✅ **Mạnh mẽ**: Pretrained components, LFM2 fusion
✅ **Linh hoạt**: Có thể dùng pretrained hoặc train from scratch
✅ **Scalable**: Có thể điều chỉnh số layers, dimensions

**Expected Performance trên RAVDESS:**
- Custom LFM2: ~80% accuracy
- Pretrained LFM2: ~82-85% accuracy
