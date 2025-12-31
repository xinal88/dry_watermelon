"""
Complete Visual Branch Module

Combines:
1. SigLIP Vision Encoder - Extract patch tokens from frames
2. ROI Token Compression - Compress to important tokens
3. Temporal Encoder - Model temporal dynamics

Pipeline:
    Video [B, T, C, H, W]
    -> SigLIP Encoder [B, T, N, D]
    -> ROI Compression [B, T, K, D]
    -> Temporal Encoder [B, S, D]
    -> Visual Features
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

from .siglip_encoder import SigLIPEncoder
from .roi_compression import ROITokenCompression
from .temporal_encoder import TemporalEncoder


@dataclass
class VisualBranchConfig:
    """Configuration for Visual Branch."""

    # SigLIP2 Encoder (upgraded from SigLIP1)
    pretrained_model: str = "google/siglip2-base-patch16-224"
    use_pretrained_encoder: bool = True  # Load SigLIP2 by default
    feature_dim: int = 768
    freeze_encoder: bool = False
    encoder_backend: str = "transformers"
    
    # ROI Compression
    num_keep_tokens: int = 64
    num_global_tokens: int = 4
    roi_weight: float = 2.0
    compression_temperature: float = 1.0
    
    # Temporal Encoder
    temporal_depth: int = 6
    temporal_heads: int = 8
    gscb_ratio: float = 0.7
    gscb_kernel_size: int = 4
    num_segments: int = 8
    
    # General
    dropout: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pretrained_model": self.pretrained_model,
            "feature_dim": self.feature_dim,
            "freeze_encoder": self.freeze_encoder,
            "encoder_backend": self.encoder_backend,
            "num_keep_tokens": self.num_keep_tokens,
            "num_global_tokens": self.num_global_tokens,
            "roi_weight": self.roi_weight,
            "compression_temperature": self.compression_temperature,
            "temporal_depth": self.temporal_depth,
            "temporal_heads": self.temporal_heads,
            "gscb_ratio": self.gscb_ratio,
            "gscb_kernel_size": self.gscb_kernel_size,
            "num_segments": self.num_segments,
            "dropout": self.dropout,
        }


class VisualBranch(nn.Module):
    """
    Complete Visual Branch for video-based emotion recognition.
    
    Processes video frames through:
    1. SigLIP encoder for patch features
    2. ROI-aware token compression
    3. Hybrid temporal encoder (GSCB + Attention)
    """
    
    def __init__(
        self,
        pretrained_model: str = "google/siglip2-base-patch16-224",
        use_pretrained_encoder: bool = False,
        feature_dim: int = 768,
        freeze_encoder: bool = False,
        encoder_backend: str = "transformers",
        num_keep_tokens: int = 64,
        num_global_tokens: int = 4,
        roi_weight: float = 2.0,
        compression_temperature: float = 1.0,
        temporal_depth: int = 6,
        temporal_heads: int = 8,
        gscb_ratio: float = 0.7,
        gscb_kernel_size: int = 4,
        num_segments: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_segments = num_segments
        self.use_pretrained_encoder = use_pretrained_encoder
        
        # 1. Vision Encoder (SigLIP2 or Custom)
        self.encoder = SigLIPEncoder(
            pretrained_model=pretrained_model,
            feature_dim=feature_dim,
            freeze_encoder=freeze_encoder,
            backend=encoder_backend,
            use_pretrained=use_pretrained_encoder,
        )
        
        # 2. ROI Token Compression
        self.roi_compression = ROITokenCompression(
            input_dim=feature_dim,
            num_keep_tokens=num_keep_tokens,
            num_global_tokens=num_global_tokens,
            roi_weight=roi_weight,
            temperature=compression_temperature,
        )
        
        # 3. Temporal Encoder
        self.temporal_encoder = TemporalEncoder(
            dim=feature_dim,
            depth=temporal_depth,
            num_heads=temporal_heads,
            gscb_ratio=gscb_ratio,
            kernel_size=gscb_kernel_size,
            dropout=dropout,
            num_segments=num_segments,
        )
        
        print(f"\n[OK] Visual Branch initialized:")
        print(f"  - Encoder: {pretrained_model}")
        print(f"  - Token compression: {num_keep_tokens} + {num_global_tokens} tokens")
        print(f"  - Temporal: {temporal_depth} layers, {num_segments} segments")
    
    @classmethod
    def from_config(cls, config: VisualBranchConfig) -> "VisualBranch":
        """Create from config object."""
        return cls(**config.to_dict())
    
    def forward(
        self,
        frames: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for video frames.
        
        Args:
            frames: [B, T, C, H, W] video frames
            roi_mask: [B, T, N] optional ROI mask for patches
            return_intermediates: Return intermediate features
            
        Returns:
            {
                "segment_features": [B, S, D] - Segment-level features
                "frame_features": [B, T, D] - Frame-level features
                "patch_tokens": [B, T, N, D] - (optional) Raw patch tokens
                "compressed_tokens": [B, T, K, D] - (optional) Compressed tokens
            }
        """
        # 1. Encode frames with SigLIP
        encoder_output = self.encoder(frames)
        patch_tokens = encoder_output["patch_tokens"]  # [B, T, N, D]

        # 2. ROI-aware token compression
        compression_output = self.roi_compression(
            patch_tokens,
            roi_mask=roi_mask,
            return_scores=return_intermediates,
        )
        compressed_tokens = compression_output["compressed_tokens"]  # [B, T, K+G, D]

        # 3. Temporal encoding
        temporal_output = self.temporal_encoder(compressed_tokens)
        segment_features = temporal_output["segment_features"]  # [B, S, D]
        frame_features = temporal_output["frame_features"]  # [B, T, D]

        # Build output
        result = {
            "segment_features": segment_features,
            "frame_features": frame_features,
        }

        if return_intermediates:
            result["patch_tokens"] = patch_tokens
            result["compressed_tokens"] = compressed_tokens
            if "scores" in compression_output:
                result["importance_scores"] = compression_output["scores"]

        return result

    def get_segment_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Convenience method to get only segment features."""
        output = self.forward(frames, return_intermediates=False)
        return output["segment_features"]

    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """Count parameters by component."""
        def count(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": count(self.encoder),
            "roi_compression": count(self.roi_compression),
            "temporal_encoder": count(self.temporal_encoder),
            "total": count(self),
        }

    def print_summary(self):
        """Print model summary."""
        params = self.count_parameters()
        print("\n" + "="*50)
        print("Visual Branch Summary")
        print("="*50)
        print(f"Encoder:          {params['encoder']:>12,} params")
        print(f"ROI Compression:  {params['roi_compression']:>12,} params")
        print(f"Temporal Encoder: {params['temporal_encoder']:>12,} params")
        print("-"*50)
        print(f"Total:            {params['total']:>12,} params")
        print(f"                  ({params['total']/1e6:.2f}M)")
        print("="*50)
