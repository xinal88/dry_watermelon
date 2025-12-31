"""
Custom LFM2 Layer Implementation
Fallback implementation if pretrained model cannot be loaded.

Based on LFM2 architecture:
- Lfm2ShortConv: Gated short convolution for local patterns
- Lfm2Attention: Multi-head attention for global dependencies
- Lfm2MLP: Feed-forward network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class Lfm2RMSNorm(nn.Module):
    """RMS Normalization used in LFM2."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class Lfm2ShortConv(nn.Module):
    """
    Gated Short Convolution Block from LFM2.
    Uses depthwise convolution with gating mechanism.
    """
    
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        expansion_factor: float = 3.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        expanded_size = int(hidden_size * expansion_factor)
        
        # Input projection (to B, C, x gates)
        self.in_proj = nn.Linear(hidden_size, expanded_size, bias=False)
        
        # Depthwise convolution
        self.conv = nn.Conv1d(
            expanded_size,
            expanded_size,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size - 1,  # Causal padding
            groups=expanded_size,  # Depthwise
            bias=False,
        )
        
        # Output projection
        self.out_proj = nn.Linear(expanded_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        
        # Project and split for gating
        hidden = self.in_proj(x)  # [B, T, expanded]
        
        # Apply gating (SiLU activation)
        hidden = F.silu(hidden)
        
        # Convolution (need [B, C, T] format)
        hidden = hidden.transpose(1, 2)  # [B, expanded, T]
        hidden = self.conv(hidden)
        hidden = hidden[:, :, :T]  # Remove causal padding
        hidden = hidden.transpose(1, 2)  # [B, T, expanded]
        
        # Output projection
        output = self.out_proj(hidden)
        
        return output


class Lfm2Attention(nn.Module):
    """
    Multi-head attention from LFM2.
    Uses grouped query attention (GQA) for efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 24,
        num_kv_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layer norms for Q and K
        self.q_layernorm = Lfm2RMSNorm(self.head_dim)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attention_mask: [B, T] or [B, 1, T, T]
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x)  # [B, T, D]
        K = self.k_proj(x)  # [B, T, num_kv_heads * head_dim]
        V = self.v_proj(x)  # [B, T, num_kv_heads * head_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_kv_heads, self.head_dim)
        V = V.view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply layer norm to Q and K
        Q = self.q_layernorm(Q)
        K = self.k_layernorm(K)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [B, num_heads, T, head_dim]
        K = K.transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        V = V.transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        
        # Repeat K and V for grouped query attention
        if self.num_kv_groups > 1:
            K = K.repeat_interleave(self.num_kv_groups, dim=1)
            V = V.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()  # [B, T, num_heads, head_dim]
        output = output.view(B, T, D)
        output = self.out_proj(output)
        
        return output


class Lfm2MLP(nn.Module):
    """
    Feed-forward network from LFM2.
    Uses SwiGLU activation (similar to LLaMA).
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 4.5)  # LFM2 uses 4.5x
        
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation: SiLU(W1(x)) * W3(x)
        
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LFM2DecoderLayer(nn.Module):
    """
    Single LFM2 Decoder Layer.
    Can use either ShortConv or Attention as the operator.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 24,
        num_kv_heads: int = 8,
        dropout: float = 0.1,
        use_conv: bool = True,
    ):
        super().__init__()
        
        self.use_conv = use_conv
        
        # Operator (Conv or Attention)
        if use_conv:
            self.operator = Lfm2ShortConv(hidden_dim, kernel_size=3)
        else:
            self.operator = Lfm2Attention(
                hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
            )
        
        # Feed-forward network
        self.feed_forward = Lfm2MLP(hidden_dim)
        
        # Layer norms
        self.operator_norm = Lfm2RMSNorm(hidden_dim)
        self.ffn_norm = Lfm2RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attention_mask: Optional mask for attention
        Returns:
            [B, T, D]
        """
        # Operator (Conv or Attention) with residual
        residual = x
        x = self.operator_norm(x)
        if self.use_conv:
            x = self.operator(x)
        else:
            x = self.operator(x, attention_mask)
        x = residual + x
        
        # Feed-forward with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x
