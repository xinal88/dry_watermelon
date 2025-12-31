"""
Complete Multimodal FER Model

Integrates:
1. Audio Branch (FastConformer + Segment Pooling)
2. Visual Branch (SigLIP + ROI + Temporal Encoder)
3. LFM2 Fusion (Liquid Neural Network)
4. Emotion Classifier

Pipeline:
    Audio [B, T_audio] ──> Audio Branch ──> [B, S, 512] ──┐
                                                            ├──> LFM2 Fusion ──> [B, S, 512] ──> Classifier ──> [B, 8]
    Video [B, T, 3, H, W] ──> Visual Branch ──> [B, S, 768] ──┘
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .audio_branch import AudioBranch, AudioBranchConfig
from .visual_branch import VisualBranch, VisualBranchConfig
from .fusion import LFM2Fusion, LFM2FusionConfig
from .classifier import EmotionClassifier, ClassifierConfig


@dataclass
class MultimodalFERConfig:
    """Complete configuration for Multimodal FER model."""
    
    # Audio branch
    audio_config: AudioBranchConfig = None
    
    # Visual branch
    visual_config: VisualBranchConfig = None
    
    # Fusion
    fusion_config: LFM2FusionConfig = None
    
    # Classifier
    classifier_config: ClassifierConfig = None
    
    # General
    num_classes: int = 8  # RAVDESS emotions
    num_segments: int = 8
    
    def __post_init__(self):
        # Initialize default configs if not provided
        if self.audio_config is None:
            self.audio_config = AudioBranchConfig(
                feature_dim=512,
                num_segments=self.num_segments,
            )
        
        if self.visual_config is None:
            self.visual_config = VisualBranchConfig(
                feature_dim=768,
                num_segments=self.num_segments,
            )
        
        if self.fusion_config is None:
            self.fusion_config = LFM2FusionConfig(
                audio_dim=512,
                visual_dim=768,
                num_segments=self.num_segments,
                output_dim=512,
            )
        
        if self.classifier_config is None:
            self.classifier_config = ClassifierConfig(
                input_dim=512,
                num_classes=self.num_classes,
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_config": self.audio_config,
            "visual_config": self.visual_config,
            "fusion_config": self.fusion_config,
            "classifier_config": self.classifier_config,
            "num_classes": self.num_classes,
            "num_segments": self.num_segments,
        }


class MultimodalFER(nn.Module):
    """
    Complete Multimodal Facial Expression Recognition Model.
    
    Combines audio and visual modalities for emotion recognition.
    """
    
    def __init__(
        self,
        audio_config: AudioBranchConfig = None,
        visual_config: VisualBranchConfig = None,
        fusion_config: LFM2FusionConfig = None,
        classifier_config: ClassifierConfig = None,
        num_classes: int = 8,
        num_segments: int = 8,
    ):
        super().__init__()
        
        # Create default config
        config = MultimodalFERConfig(
            audio_config=audio_config,
            visual_config=visual_config,
            fusion_config=fusion_config,
            classifier_config=classifier_config,
            num_classes=num_classes,
            num_segments=num_segments,
        )
        
        self.num_classes = num_classes
        self.num_segments = num_segments
        
        # Build components
        print("\n" + "="*70)
        print("Building Multimodal FER Model")
        print("="*70)
        
        print("\n[1/4] Audio Branch...")
        self.audio_branch = AudioBranch(**config.audio_config.to_dict())
        
        print("\n[2/4] Visual Branch...")
        self.visual_branch = VisualBranch(**config.visual_config.to_dict())
        
        print("\n[3/4] LFM2 Fusion...")
        self.fusion = LFM2Fusion(**config.fusion_config.to_dict())
        
        print("\n[4/4] Emotion Classifier...")
        self.classifier = EmotionClassifier(**config.classifier_config.to_dict())
        
        print("\n" + "="*70)
        print("✅ Multimodal FER Model Built Successfully!")
        print("="*70)
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            audio: [B, T_audio] - Raw audio waveform
            video: [B, T_frames, 3, H, W] - Video frames
            audio_lengths: [B] - Actual audio lengths
            roi_mask: [B, T_frames, N_patches] - ROI mask for face regions
            return_intermediates: Return intermediate features
            
        Returns:
            {
                "logits": [B, num_classes] - Emotion logits
                "probabilities": [B, num_classes] - Emotion probabilities
                "audio_features": [B, S, D_audio] - (optional)
                "visual_features": [B, S, D_visual] - (optional)
                "fused_features": [B, S, D_fused] - (optional)
            }
        """
        # 1. Audio Branch
        audio_output = self.audio_branch(
            audio,
            audio_lengths=audio_lengths,
            return_all_segments=True,
        )
        audio_features = audio_output["segment_features"]  # [B, S, 512]
        
        # 2. Visual Branch
        visual_output = self.visual_branch(
            video,
            roi_mask=roi_mask,
            return_intermediates=False,
        )
        visual_features = visual_output["segment_features"]  # [B, S, 768]
        
        # 3. Fusion
        fusion_output = self.fusion(
            audio_features,
            visual_features,
            return_intermediates=return_intermediates,
        )
        fused_features = fusion_output["fused_features"]  # [B, S, 512]
        
        # 4. Classification
        classifier_output = self.classifier(
            fused_features,
            return_features=return_intermediates,
        )
        
        result = {
            "logits": classifier_output["logits"],
            "probabilities": classifier_output["probabilities"],
        }
        
        if return_intermediates:
            result["audio_features"] = audio_features
            result["visual_features"] = visual_features
            result["fused_features"] = fused_features
            result["pooled_features"] = fusion_output["pooled_features"]
        
        return result
    
    def forward_audio_only(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with audio only (for ablation studies)."""
        audio_output = self.audio_branch(audio, audio_lengths, return_all_segments=False)
        audio_features = audio_output["segment_features"]  # [B, 512]
        
        # Direct classification from audio
        classifier_output = self.classifier(audio_features)
        
        return {
            "logits": classifier_output["logits"],
            "probabilities": classifier_output["probabilities"],
        }
    
    def forward_visual_only(
        self,
        video: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with video only (for ablation studies)."""
        visual_output = self.visual_branch(video, roi_mask, return_intermediates=False)
        
        # Need to project visual features to classifier input dim
        visual_features = visual_output["segment_features"]  # [B, S, 768]
        
        # Simple projection
        if not hasattr(self, 'visual_proj'):
            self.visual_proj = nn.Linear(768, 512).to(visual_features.device)
        
        visual_features = self.visual_proj(visual_features)  # [B, S, 512]
        
        classifier_output = self.classifier(visual_features)
        
        return {
            "logits": classifier_output["logits"],
            "probabilities": classifier_output["probabilities"],
        }
    
    @classmethod
    def from_config(cls, config: MultimodalFERConfig) -> "MultimodalFER":
        """Create model from config."""
        return cls(**config.to_dict())
    
    def count_parameters(self, by_component: bool = True) -> Dict[str, int]:
        """Count model parameters."""
        if by_component:
            return {
                "audio_branch": sum(p.numel() for p in self.audio_branch.parameters()),
                "visual_branch": sum(p.numel() for p in self.visual_branch.parameters()),
                "fusion": sum(p.numel() for p in self.fusion.parameters()),
                "classifier": sum(p.numel() for p in self.classifier.parameters()),
                "total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            }
        else:
            return {
                "total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            }
    
    def print_summary(self):
        """Print detailed model summary."""
        params = self.count_parameters(by_component=True)
        
        print("\n" + "="*70)
        print("Multimodal FER Model Summary")
        print("="*70)
        print(f"Audio Branch:      {params['audio_branch']:>15,} params ({params['audio_branch']/1e6:>6.2f}M)")
        print(f"Visual Branch:     {params['visual_branch']:>15,} params ({params['visual_branch']/1e6:>6.2f}M)")
        print(f"LFM2 Fusion:       {params['fusion']:>15,} params ({params['fusion']/1e6:>6.2f}M)")
        print(f"Classifier:        {params['classifier']:>15,} params ({params['classifier']/1e6:>6.2f}M)")
        print("-"*70)
        print(f"Total:             {params['total']:>15,} params ({params['total']/1e6:>6.2f}M)")
        print(f"Trainable:         {params['trainable']:>15,} params ({params['trainable']/1e6:>6.2f}M)")
        print("="*70)
        
        # Memory estimation
        memory_fp32 = params['total'] * 4 / 1e9
        memory_fp16 = params['total'] * 2 / 1e9
        print(f"\nEstimated Memory:")
        print(f"  FP32: {memory_fp32:.2f} GB")
        print(f"  FP16/BF16: {memory_fp16:.2f} GB")
        print("="*70)
    
    def get_emotion_labels(self) -> list:
        """Get RAVDESS emotion labels."""
        return [
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ]
