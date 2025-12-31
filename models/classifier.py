"""
Emotion Classifier Head

Simple MLP classifier for emotion recognition from fused features.
Supports multiple pooling strategies and regularization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    """Configuration for Classifier Head."""
    
    input_dim: int = 512
    hidden_dims: list = None  # [512, 256]
    num_classes: int = 8  # RAVDESS emotions
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    use_batch_norm: bool = False
    pooling_type: Literal["mean", "max", "attention", "last"] = "mean"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]
    
    def to_dict(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "pooling_type": self.pooling_type,
        }


class TemporalPooling(nn.Module):
    """
    Temporal pooling strategies for segment features.
    """
    
    def __init__(
        self,
        pooling_type: str = "mean",
        input_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            # Learnable attention pooling
            self.query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] - Segment features
        Returns:
            [B, D] - Pooled features
        """
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        
        elif self.pooling_type == "last":
            return x[:, -1, :]
        
        elif self.pooling_type == "attention":
            B = x.shape[0]
            query = self.query.expand(B, -1, -1)
            pooled, _ = self.attention(query, x, x)
            return pooled.squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class EmotionClassifier(nn.Module):
    """
    Emotion Classifier Head.
    
    Takes fused multimodal features and predicts emotion class.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = None,
        num_classes: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = False,
        pooling_type: str = "mean",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        
        # Temporal pooling (if input has sequence dimension)
        self.temporal_pool = TemporalPooling(
            pooling_type=pooling_type,
            input_dim=input_dim,
        )
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"\n[OK] Emotion Classifier initialized:")
        print(f"  - Input: {input_dim}")
        print(f"  - Hidden: {hidden_dims}")
        print(f"  - Classes: {num_classes}")
        print(f"  - Pooling: {pooling_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, D] or [B, S, D] - Input features
            return_features: Return intermediate features
            
        Returns:
            {
                "logits": [B, num_classes] - Class logits
                "probabilities": [B, num_classes] - Class probabilities
                "features": [B, D] - (optional) Pooled features
            }
        """
        # Handle sequence input
        if x.dim() == 3:
            # [B, S, D] -> [B, D]
            pooled = self.temporal_pool(x)
        else:
            # [B, D]
            pooled = x
        
        # Classify
        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probabilities": probabilities,
        }
        
        if return_features:
            result["features"] = pooled
        
        return result
    
    @classmethod
    def from_config(cls, config: ClassifierConfig) -> "EmotionClassifier":
        """Create from config object."""
        return cls(**config.to_dict())
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_summary(self):
        """Print model summary."""
        params = self.count_parameters()
        print("\n" + "="*50)
        print("Emotion Classifier Summary")
        print("="*50)
        print(f"Total Parameters: {params:,} ({params/1e6:.2f}M)")
        print("="*50)
