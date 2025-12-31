"""
Training Module

Contains:
- Loss functions (CrossEntropy, Auxiliary, Contrastive)
- Metrics (UAR, WAR, WA-F1)
- Training utilities
"""

from .losses import EmotionLoss, MultimodalLoss, ContrastiveLoss
from .metrics import EmotionMetrics, compute_metrics_from_outputs

__all__ = [
    "EmotionLoss",
    "MultimodalLoss",
    "ContrastiveLoss",
    "EmotionMetrics",
    "compute_metrics_from_outputs",
]
