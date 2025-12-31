"""
Temporal Encoder for Video Understanding

Implements hybrid temporal modeling with:
1. GSCB (Gated Short Convolution Block) - Local temporal patterns
2. Attention layers - Global temporal dependencies
3. Optional Mamba2 integration - Efficient long-range modeling

Inspired by LFM2 architecture: 70% GSCB + 30% Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class GatedShortConvBlock(nn.Module):
    """
    Gated Short Convolution Block (GSCB).
    
    Inspired by LFM2's short convolution layers for local pattern modeling.
    Efficient for capturing micro-expressions and local temporal dynamics.
    
    Architecture:
        x -> Linear -> B, C gates -> ShortConv -> Gating -> Linear -> output
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        expansion_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        hidden_dim = int(dim * expansion_factor)
        
        # Input projection with gating
        self.input_proj = nn.Linear(dim, hidden_dim * 3)  # B, C, x
        
        # Short convolution (causal)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Causal padding
            groups=hidden_dim,  # Depthwise
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, T, D] temporal features
            mask: [B, T] optional attention mask
            
        Returns:
            output: [B, T, D] processed features
        """
        residual = x
        x = self.norm(x)
        
        # Input projection with gating
        B, T, D = x.shape
        proj = self.input_proj(x)  # [B, T, 3*hidden]
        gate_b, gate_c, hidden = proj.chunk(3, dim=-1)
        
        # Apply first gate
        hidden = hidden * torch.sigmoid(gate_b)
        
        # Short convolution (need [B, C, T] format)
        hidden = hidden.transpose(1, 2)  # [B, hidden, T]
        hidden = self.conv(hidden)
        hidden = hidden[:, :, :T]  # Remove causal padding
        hidden = hidden.transpose(1, 2)  # [B, T, hidden]
        
        # Apply second gate
        hidden = hidden * torch.sigmoid(gate_c)
        
        # Output projection
        output = self.output_proj(hidden)
        output = self.dropout(output)
        
        return output + residual


class TemporalAttentionBlock(nn.Module):
    """
    Temporal attention block for global dependencies.
    Uses multi-head self-attention across temporal dimension.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, T, D] temporal features
            mask: [B, T] optional attention mask
            
        Returns:
            output: [B, T, D] processed features
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + attn_out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class TemporalEncoder(nn.Module):
    """
    Hybrid Temporal Encoder combining GSCB and Attention.

    Architecture inspired by LFM2:
    - 70% GSCB layers for local temporal patterns
    - 30% Attention layers for global dependencies

    Processes: [B, T, K, D] -> [B, S, D] (temporal segments)
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        gscb_ratio: float = 0.7,
        kernel_size: int = 4,
        dropout: float = 0.1,
        num_segments: int = 8,
        pooling_type: str = "attention",
    ):
        """
        Args:
            dim: Feature dimension
            depth: Number of layers
            num_heads: Attention heads
            gscb_ratio: Ratio of GSCB layers (0.7 = 70%)
            kernel_size: GSCB conv kernel size
            dropout: Dropout rate
            num_segments: Output temporal segments
            pooling_type: "attention", "avg", or "learned"
        """
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.num_segments = num_segments

        # Determine layer types
        num_gscb = int(depth * gscb_ratio)
        num_attn = depth - num_gscb

        # Create layers with interleaved pattern
        self.layers = nn.ModuleList()
        gscb_positions = self._get_layer_positions(depth, num_gscb)

        for i in range(depth):
            if i in gscb_positions:
                self.layers.append(
                    GatedShortConvBlock(dim, kernel_size, dropout=dropout)
                )
            else:
                self.layers.append(
                    TemporalAttentionBlock(dim, num_heads, dropout=dropout)
                )

        # Spatial pooling (pool tokens within each frame)
        self.spatial_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

        # Temporal segment pooling
        self.segment_pool = self._build_segment_pooling(pooling_type, dim)

        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)

        print(f"[OK] TemporalEncoder: {num_gscb} GSCB + {num_attn} Attention layers")

    def _get_layer_positions(self, total: int, num_type: int) -> set:
        """Get positions for GSCB layers (evenly distributed)."""
        if num_type == 0:
            return set()
        if num_type >= total:
            return set(range(total))

        # Distribute evenly
        step = total / num_type
        positions = {int(i * step) for i in range(num_type)}
        return positions

    def _build_segment_pooling(self, pooling_type: str, dim: int) -> nn.Module:
        """Build segment pooling module."""
        if pooling_type == "attention":
            return nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
        elif pooling_type == "learned":
            # Learnable segment queries
            return nn.Parameter(torch.randn(1, self.num_segments, dim) * 0.02)
        else:
            return None  # Use avg pooling

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, T, K, D] - Compressed tokens per frame
                B: batch, T: frames, K: tokens per frame, D: dim
            temporal_mask: [B, T] - Optional mask for frames

        Returns:
            {
                "segment_features": [B, S, D] - Segment-level features
                "frame_features": [B, T, D] - Frame-level features
            }
        """
        B, T, K, D = x.shape

        # 1. Pool tokens within each frame -> [B, T, D]
        # Compute attention weights
        attn_weights = self.spatial_pool(x).squeeze(-1)  # [B, T, K]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        frame_features = torch.einsum('btk,btkd->btd', attn_weights, x)  # [B, T, D]

        # 2. Apply temporal layers
        for layer in self.layers:
            frame_features = layer(frame_features, mask=temporal_mask)

        frame_features = self.final_norm(frame_features)

        # 3. Segment pooling -> [B, S, D]
        segment_features = self._segment_pool(frame_features, temporal_mask)

        return {
            "segment_features": segment_features,  # [B, S, D]
            "frame_features": frame_features,  # [B, T, D]
        }

    def _segment_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool temporal features into segments."""
        B, T, D = x.shape
        S = self.num_segments

        if isinstance(self.segment_pool, nn.MultiheadAttention):
            # Learnable segment queries
            queries = torch.zeros(B, S, D, device=x.device, dtype=x.dtype)
            # Initialize with temporal positions
            for s in range(S):
                start = int(s * T / S)
                end = int((s + 1) * T / S)
                queries[:, s] = x[:, start:end].mean(dim=1)

            # Cross-attention
            segment_features, _ = self.segment_pool(queries, x, x)

        elif isinstance(self.segment_pool, nn.Parameter):
            # Use learned queries
            queries = self.segment_pool.expand(B, S, D)
            # Simple averaging for now (can add cross-attention)
            segment_size = T // S
            segment_features = []
            for s in range(S):
                start = s * segment_size
                end = start + segment_size if s < S - 1 else T
                segment_features.append(x[:, start:end].mean(dim=1))
            segment_features = torch.stack(segment_features, dim=1)

        else:
            # Average pooling per segment
            segment_size = T // S
            segment_features = []
            for s in range(S):
                start = s * segment_size
                end = start + segment_size if s < S - 1 else T
                segment_features.append(x[:, start:end].mean(dim=1))
            segment_features = torch.stack(segment_features, dim=1)

        return segment_features

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
