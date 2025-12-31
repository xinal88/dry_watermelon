"""
Custom Lightweight Conformer Implementation
Based on: "Conformer: Convolution-augmented Transformer for Speech Recognition"

Conformer Block = Feed Forward + Multi-Head Self-Attention + Convolution + Feed Forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConformerBlock(nn.Module):
    """
    Single Conformer block with:
    1. Feed-forward module (1/2)
    2. Multi-head self-attention
    3. Convolution module
    4. Feed-forward module (2/2)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feed-forward modules (Macaron-style)
        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        
        # Multi-head self-attention
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Layer norms
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_mha = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)
        
        # Final layer norm
        self.norm_out = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] - Padding mask
        Returns:
            output: [B, T, D]
        """
        # 1. First feed-forward (half-step residual)
        residual = x
        x = self.norm_ff1(x)
        x = residual + 0.5 * self.dropout(self.ff1(x))
        
        # 2. Multi-head self-attention
        residual = x
        x = self.norm_mha(x)
        x = residual + self.dropout(self.mha(x, mask))
        
        # 3. Convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv(x))
        
        # 4. Second feed-forward (half-step residual)
        residual = x
        x = self.norm_ff2(x)
        x = residual + 0.5 * self.dropout(self.ff2(x))
        
        # Final layer norm
        x = self.norm_out(x)
        
        return x


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward network with Swish activation"""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3*D]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D/H]
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class ConvolutionModule(nn.Module):
    """Convolution module with gating mechanism"""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        
        # Pointwise conv 1 + GLU
        x = self.pointwise_conv1(x)  # [B, 2*D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        
        # Depthwise conv + batch norm + activation
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, T, D]

        return x


class ConformerEncoder(nn.Module):
    """Full Conformer encoder with multiple blocks"""

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 17,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, input_dim] - Input features (e.g., mel spectrogram)
            lengths: [B] - Actual sequence lengths
        Returns:
            output: [B, T, d_model]
            lengths: [B]
        """
        # Input projection
        x = self.input_proj(x)
        x = self.dropout(x)

        # Create padding mask if lengths provided
        mask = None
        if lengths is not None:
            B, T = x.shape[:2]
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Pass through Conformer blocks
        for block in self.blocks:
            x = block(x, mask)

        return x, lengths


class ConformerEncoder(nn.Module):
    """Full Conformer encoder with multiple blocks"""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 17,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, input_dim] - Input features (e.g., mel spectrogram)
            lengths: [B] - Actual sequence lengths
        Returns:
            output: [B, T, d_model]
            lengths: [B]
        """
        # Input projection
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Create padding mask if lengths provided
        mask = None
        if lengths is not None:
            B, T = x.shape[:2]
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Pass through Conformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        return x, lengths

