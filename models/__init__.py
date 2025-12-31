"""
Models package for Lightweight Multimodal FER
"""

from .audio_branch import AudioBranch, AudioBranchConfig
from .visual_branch import (
    VisualBranch,
    VisualBranchConfig,
    SigLIPEncoder,
    ROITokenCompression,
    TemporalEncoder,
)
from .fusion import LFM2Fusion, LFM2FusionConfig
from .classifier import EmotionClassifier, ClassifierConfig
from .multimodal_fer import MultimodalFER, MultimodalFERConfig

__all__ = [
    # Audio Branch
    "AudioBranch",
    "AudioBranchConfig",
    # Visual Branch
    "VisualBranch",
    "VisualBranchConfig",
    "SigLIPEncoder",
    "ROITokenCompression",
    "TemporalEncoder",
    # Fusion
    "LFM2Fusion",
    "LFM2FusionConfig",
    # Classifier
    "EmotionClassifier",
    "ClassifierConfig",
    # Complete Model
    "MultimodalFER",
    "MultimodalFERConfig",
]

