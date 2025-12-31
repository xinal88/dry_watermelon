"""
Visual Branch Module
SigLIP Encoder + ROI-aware Token Compression + Temporal Encoder

Pipeline:
    Video Frames [B, T, C, H, W]
    -> SigLIP Encoder [B, T, N, D] (N = num patches)
    -> ROI Token Compression [B, T, K, D] (K << N)
    -> Temporal Encoder [B, S, D]
    -> Output: Segment-level visual features
"""

from .siglip_encoder import SigLIPEncoder
from .roi_compression import ROITokenCompression
from .temporal_encoder import TemporalEncoder
from .visual_branch import VisualBranch, VisualBranchConfig

__all__ = [
    "SigLIPEncoder",
    "ROITokenCompression", 
    "TemporalEncoder",
    "VisualBranch",
    "VisualBranchConfig",
]

