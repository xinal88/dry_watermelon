"""
Audio Branch Module
FastConformer + Segment Attention Pooling
"""

from .audio_branch import AudioBranch, AudioBranchConfig
from .fastconformer import FastConformerEncoder, AudioPreprocessor
from .segment_pooling import SegmentAttentionPooling, AttentionPooling

__all__ = [
    "AudioBranch",
    "AudioBranchConfig",
    "FastConformerEncoder",
    "AudioPreprocessor",
    "SegmentAttentionPooling",
    "AttentionPooling",
]

