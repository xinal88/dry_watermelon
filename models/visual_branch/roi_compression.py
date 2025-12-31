"""
ROI-aware Token Compression Module

Implements STORM-style token compression with:
1. ROI-biased scoring (focus on facial regions)
2. Gumbel Top-K differentiable selection
3. Global tokens for context

Pipeline:
    Patch Tokens [B, T, N, D] + ROI Mask
    -> Importance Scoring [B, T, N]
    -> Gumbel Top-K Selection [B, T, K]
    -> Selected Tokens + Global Tokens [B, T, K+G, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class ROITokenCompression(nn.Module):
    """
    ROI-aware token compression with differentiable selection.
    
    Compresses N patch tokens to K tokens based on:
    1. Learned importance scores
    2. ROI (Region of Interest) bias for facial regions
    3. Gumbel-Softmax for differentiable top-k selection
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_keep_tokens: int = 64,
        num_global_tokens: int = 4,
        roi_weight: float = 2.0,
        temperature: float = 1.0,
        hard_selection: bool = True,
        use_learned_scoring: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_keep_tokens: Number of tokens to keep (K)
            num_global_tokens: Number of global context tokens (G)
            roi_weight: Weight multiplier for ROI regions
            temperature: Gumbel-Softmax temperature
            hard_selection: Use hard (straight-through) selection
            use_learned_scoring: Use learned scoring vs simple attention
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_keep_tokens = num_keep_tokens
        self.num_global_tokens = num_global_tokens
        self.roi_weight = roi_weight
        self.temperature = temperature
        self.hard_selection = hard_selection
        
        # Importance scoring network
        if use_learned_scoring:
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
            )
        else:
            # Simple attention-based scoring
            self.scorer = nn.Linear(input_dim, 1)
        
        # Global tokens (learnable)
        if num_global_tokens > 0:
            self.global_tokens = nn.Parameter(
                torch.randn(1, 1, num_global_tokens, input_dim) * 0.02
            )
        else:
            self.global_tokens = None
        
        # Cross-attention for global tokens to aggregate context
        if num_global_tokens > 0:
            self.global_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
    
    def compute_importance_scores(
        self,
        tokens: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute importance scores for each token.
        
        Args:
            tokens: [B, T, N, D] patch tokens
            roi_mask: [B, T, N] binary mask (1 = ROI, 0 = background)
            
        Returns:
            scores: [B, T, N] importance scores
        """
        B, T, N, D = tokens.shape
        
        # Compute base scores
        tokens_flat = tokens.view(B * T, N, D)
        scores = self.scorer(tokens_flat).squeeze(-1)  # [B*T, N]
        scores = scores.view(B, T, N)  # [B, T, N]
        
        # Apply ROI bias
        if roi_mask is not None:
            # Boost scores for ROI regions
            roi_bias = (roi_mask.float() * (self.roi_weight - 1.0)) + 1.0
            scores = scores * roi_bias
        
        return scores
    
    def gumbel_top_k(
        self,
        scores: torch.Tensor,
        k: int,
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable top-k selection using Gumbel-Softmax.
        
        Args:
            scores: [B, T, N] importance scores
            k: Number of tokens to select
            tau: Temperature for Gumbel-Softmax
            hard: Use straight-through estimator
            
        Returns:
            selection_weights: [B, T, N] soft selection weights
            indices: [B, T, K] selected indices
        """
        B, T, N = scores.shape
        
        if self.training:
            # Add Gumbel noise for exploration
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1e-8)
            )).clamp(min=-10, max=10)
            perturbed_scores = scores + gumbel_noise
        else:
            perturbed_scores = scores
        
        # Get top-k indices
        _, indices = torch.topk(perturbed_scores, k, dim=-1)  # [B, T, K]
        
        # Create selection mask
        selection_mask = torch.zeros_like(scores)
        selection_mask.scatter_(-1, indices, 1.0)  # [B, T, N]
        
        if hard:
            # Straight-through estimator
            # Forward: hard selection, Backward: soft gradients
            soft_weights = F.softmax(scores / tau, dim=-1)
            selection_weights = selection_mask - soft_weights.detach() + soft_weights
        else:
            selection_weights = F.softmax(scores / tau, dim=-1) * selection_mask

        return selection_weights, indices

    def select_tokens(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select tokens based on indices.

        Args:
            tokens: [B, T, N, D] all patch tokens
            indices: [B, T, K] selected indices

        Returns:
            selected: [B, T, K, D] selected tokens
        """
        B, T, N, D = tokens.shape
        K = indices.shape[-1]

        # Expand indices for gathering
        indices_expanded = indices.unsqueeze(-1).expand(B, T, K, D)

        # Gather selected tokens
        selected = torch.gather(tokens, dim=2, index=indices_expanded)

        return selected

    def compute_global_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global context tokens via cross-attention.

        Args:
            tokens: [B, T, N, D] patch tokens

        Returns:
            global_out: [B, T, G, D] global tokens
        """
        if self.global_tokens is None:
            return None

        B, T, N, D = tokens.shape
        G = self.num_global_tokens

        # Expand global tokens for batch and time
        global_q = self.global_tokens.expand(B, T, G, D)  # [B, T, G, D]

        # Reshape for attention: [B*T, G, D] and [B*T, N, D]
        global_q_flat = global_q.view(B * T, G, D)
        tokens_flat = tokens.view(B * T, N, D)

        # Cross-attention: global tokens attend to all patch tokens
        global_out, _ = self.global_attn(
            query=global_q_flat,
            key=tokens_flat,
            value=tokens_flat,
        )

        # Reshape back
        global_out = global_out.view(B, T, G, D)

        return global_out

    def forward(
        self,
        tokens: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compress tokens with ROI-aware selection.

        Args:
            tokens: [B, T, N, D] patch tokens from vision encoder
            roi_mask: [B, T, N] binary ROI mask (optional)
            return_scores: Whether to return importance scores

        Returns:
            {
                "compressed_tokens": [B, T, K+G, D] - Selected + global tokens
                "selected_tokens": [B, T, K, D] - Only selected tokens
                "global_tokens": [B, T, G, D] - Global context tokens
                "indices": [B, T, K] - Selected token indices
                "scores": [B, T, N] - Importance scores (if return_scores)
            }
        """
        B, T, N, D = tokens.shape

        # 1. Compute importance scores
        scores = self.compute_importance_scores(tokens, roi_mask)

        # 2. Gumbel Top-K selection
        selection_weights, indices = self.gumbel_top_k(
            scores,
            k=self.num_keep_tokens,
            tau=self.temperature,
            hard=self.hard_selection,
        )

        # 3. Select tokens
        selected_tokens = self.select_tokens(tokens, indices)  # [B, T, K, D]

        # 4. Compute global tokens
        global_tokens = self.compute_global_tokens(tokens)  # [B, T, G, D]

        # 5. Concatenate selected + global tokens
        if global_tokens is not None:
            compressed_tokens = torch.cat([selected_tokens, global_tokens], dim=2)
        else:
            compressed_tokens = selected_tokens

        result = {
            "compressed_tokens": compressed_tokens,  # [B, T, K+G, D]
            "selected_tokens": selected_tokens,  # [B, T, K, D]
            "indices": indices,  # [B, T, K]
        }

        if global_tokens is not None:
            result["global_tokens"] = global_tokens

        if return_scores:
            result["scores"] = scores

        return result

    @property
    def output_tokens(self) -> int:
        """Return total number of output tokens per frame."""
        return self.num_keep_tokens + self.num_global_tokens

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
