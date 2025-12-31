"""
Segment Attention Pooling Module
Splits temporal sequence into segments and applies attention pooling per segment.

Input: [B, T, D] - Batch, Time, Dimension
Process: Split into S segments -> Attention pool per segment
Output: [B, S, D] - Batch, Segments, Dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import math


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism for a single segment.
    Uses learnable query vector to compute attention weights.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
        # Multi-head attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Input sequence
            mask: [B, T] - Optional padding mask (1 for valid, 0 for padding)
        Returns:
            pooled: [B, D] - Pooled representation
        """
        B, T, D = x.shape
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [B, 1, D]
        K = self.k_proj(x)      # [B, T, D]
        V = self.v_proj(x)      # [B, T, D]
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, T]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, T]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        pooled = torch.matmul(attn_weights, V)  # [B, H, 1, D/H]
        
        # Reshape and project
        pooled = pooled.transpose(1, 2).contiguous().view(B, 1, D)  # [B, 1, D]
        pooled = self.out_proj(pooled)  # [B, 1, D]
        
        return pooled.squeeze(1)  # [B, D]


class SegmentAttentionPooling(nn.Module):
    """
    Segment-based Attention Pooling.
    Splits temporal sequence into segments and pools each segment independently.
    
    Architecture:
        Input: [B, T, D]
        -> Split into S segments: [B, S, T/S, D]
        -> Attention pool per segment: [B, S, D]
        -> Optional: Temporal encoding across segments
        -> Output: [B, S, D]
    """
    def __init__(
        self,
        dim: int,
        num_segments: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling_type: Literal["attention", "max", "avg", "learnable"] = "attention",
        use_temporal_encoding: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_segments = num_segments
        self.pooling_type = pooling_type
        self.use_temporal_encoding = use_temporal_encoding
        
        # Pooling mechanism
        if pooling_type == "attention":
            self.pooling = AttentionPooling(dim, num_heads, dropout)
        elif pooling_type == "learnable":
            # Learnable weighted pooling
            self.pooling_weights = nn.Parameter(torch.randn(num_segments, 1, 1))
        
        # Temporal positional encoding for segments
        if use_temporal_encoding:
            self.temporal_encoding = nn.Parameter(torch.randn(1, num_segments, dim))
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Input sequence from encoder
            lengths: [B] - Optional actual lengths of sequences (for padding)
        Returns:
            segment_features: [B, S, D] - Pooled segment representations
        """
        B, T, D = x.shape
        S = self.num_segments
        
        # Calculate segment size
        segment_size = T // S
        remainder = T % S
        
        # Trim to make divisible by num_segments
        if remainder != 0:
            x = x[:, :T - remainder, :]
            T = T - remainder
            segment_size = T // S
        
        # Reshape into segments: [B, S, T/S, D]
        x_segments = x.view(B, S, segment_size, D)
        
        # Pool each segment
        pooled_segments = []
        for i in range(S):
            segment = x_segments[:, i, :, :]  # [B, T/S, D]
            
            if self.pooling_type == "attention":
                pooled = self.pooling(segment)  # [B, D]
            elif self.pooling_type == "max":
                pooled = segment.max(dim=1)[0]  # [B, D]
            elif self.pooling_type == "avg":
                pooled = segment.mean(dim=1)  # [B, D]
            elif self.pooling_type == "learnable":
                weights = F.softmax(self.pooling_weights[i], dim=0)
                pooled = (segment * weights).sum(dim=1)  # [B, D]
            
            pooled_segments.append(pooled)
        
        # Stack segments: [B, S, D]
        segment_features = torch.stack(pooled_segments, dim=1)
        
        # Add temporal positional encoding
        if self.use_temporal_encoding:
            segment_features = segment_features + self.temporal_encoding
        
        # Layer norm and dropout
        segment_features = self.layer_norm(segment_features)
        segment_features = self.dropout(segment_features)
        
        return segment_features

