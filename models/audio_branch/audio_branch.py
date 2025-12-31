"""
Complete Audio Branch Module
Combines FastConformer Encoder + Segment Attention Pooling

Pipeline:
    Raw Audio [B, T_audio] 
    -> Mel Spectrogram [B, T, n_mels]
    -> FastConformer Encoder [B, T, D]
    -> Segment Attention Pooling [B, S, D]
    -> Output: Segment-level audio features
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .fastconformer import FastConformerEncoder
from .segment_pooling import SegmentAttentionPooling


class AudioBranch(nn.Module):
    """
    Complete audio processing branch for multimodal FER.
    
    Architecture:
        Audio Waveform -> FastConformer -> Segment Pooling -> Audio Features
    """
    
    def __init__(
        self,
        # FastConformer params
        pretrained_model: Optional[str] = None,
        feature_dim: int = 512,
        freeze_encoder: bool = False,
        use_nemo: bool = True,
        # Audio preprocessing
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        # Conformer architecture
        num_layers: int = 17,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        # Segment pooling params
        num_segments: int = 8,
        pooling_type: str = "attention",
        use_temporal_encoding: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_segments = num_segments
        
        # FastConformer Encoder
        self.encoder = FastConformerEncoder(
            pretrained_model=pretrained_model,
            feature_dim=feature_dim,
            freeze_encoder=freeze_encoder,
            use_nemo=use_nemo,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Segment Attention Pooling
        self.segment_pooling = SegmentAttentionPooling(
            dim=feature_dim,
            num_segments=num_segments,
            num_heads=num_heads,
            dropout=dropout,
            pooling_type=pooling_type,
            use_temporal_encoding=use_temporal_encoding,
        )
        
        # Optional: Additional processing layers
        self.post_pooling = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        return_all_segments: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through audio branch.
        
        Args:
            audio: [B, T_audio] - Raw audio waveform or [B, T, n_mels] - Mel spectrogram
            audio_lengths: [B] - Actual lengths of audio sequences
            return_all_segments: If True, return [B, S, D]; if False, return [B, D] (mean pooled)
        
        Returns:
            Dictionary containing:
                - segment_features: [B, S, D] or [B, D] - Audio features
                - encoder_features: [B, T, D] - Raw encoder output (optional)
                - attention_weights: Attention weights from pooling (if available)
        """
        # 1. Encode audio with FastConformer
        encoder_output = self.encoder(audio, audio_lengths)
        encoder_features = encoder_output["features"]  # [B, T, D]
        encoder_lengths = encoder_output["lengths"]
        
        # 2. Segment-based pooling
        segment_features = self.segment_pooling(
            encoder_features,
            lengths=encoder_lengths
        )  # [B, S, D]
        
        # 3. Post-pooling processing
        segment_features = self.post_pooling(segment_features)  # [B, S, D]
        
        # 4. Optional: Pool across segments for single representation
        if not return_all_segments:
            pooled_features = segment_features.mean(dim=1)  # [B, D]
        else:
            pooled_features = segment_features
        
        return {
            "segment_features": pooled_features,  # [B, S, D] or [B, D]
            "encoder_features": encoder_features,  # [B, T, D]
            "num_segments": self.num_segments,
        }
    
    def get_num_params(self) -> Dict[str, int]:
        """Get number of parameters in each component"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        pooling_params = sum(p.numel() for p in self.segment_pooling.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "encoder": encoder_params,
            "pooling": pooling_params,
            "total": total_params,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class AudioBranchConfig:
    """Configuration class for Audio Branch"""
    
    def __init__(self, **kwargs):
        # FastConformer
        self.pretrained_model = kwargs.get("pretrained_model", None)
        self.feature_dim = kwargs.get("feature_dim", 512)
        self.freeze_encoder = kwargs.get("freeze_encoder", False)
        self.use_nemo = kwargs.get("use_nemo", True)
        
        # Audio preprocessing
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.n_mels = kwargs.get("n_mels", 80)
        self.n_fft = kwargs.get("n_fft", 512)
        self.hop_length = kwargs.get("hop_length", 160)
        self.win_length = kwargs.get("win_length", 400)
        
        # Conformer architecture
        self.num_layers = kwargs.get("num_layers", 17)
        self.d_model = kwargs.get("d_model", 512)
        self.num_heads = kwargs.get("num_heads", 8)
        self.dropout = kwargs.get("dropout", 0.1)
        
        # Segment pooling
        self.num_segments = kwargs.get("num_segments", 8)
        self.pooling_type = kwargs.get("pooling_type", "attention")
        self.use_temporal_encoding = kwargs.get("use_temporal_encoding", True)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

