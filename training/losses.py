"""
Loss Functions for Multimodal FER

Implements:
- CrossEntropy with Label Smoothing (Primary)
- Auxiliary Modality-Specific Losses (Optional)
- Contrastive Loss (Optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EmotionLoss(nn.Module):
    """
    Primary loss: CrossEntropy with Label Smoothing.
    
    Args:
        num_classes: Number of emotion classes
        label_smoothing: Label smoothing factor (0.0 - 0.5)
        class_weights: Optional class weights for imbalanced data
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - Model predictions
            labels: [B] - Ground truth labels
            
        Returns:
            loss: Scalar loss value
        """
        return self.criterion(logits, labels)


class MultimodalLoss(nn.Module):
    """
    Multimodal loss with auxiliary modality-specific losses.
    
    Args:
        num_classes: Number of emotion classes
        alpha_audio: Weight for audio auxiliary loss
        alpha_visual: Weight for visual auxiliary loss
        alpha_fusion: Weight for fusion loss
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        alpha_audio: float = 0.3,
        alpha_visual: float = 0.3,
        alpha_fusion: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.alpha_audio = alpha_audio
        self.alpha_visual = alpha_visual
        self.alpha_fusion = alpha_fusion
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Auxiliary classifiers
        self.audio_classifier = nn.Linear(512, num_classes)
        self.visual_classifier = nn.Linear(768, num_classes)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dict with:
                - logits: [B, num_classes]
                - audio_features: [B, S, 512] (optional)
                - visual_features: [B, S, 768] (optional)
            labels: [B] - Ground truth labels
            
        Returns:
            Dict with total loss and component losses
        """
        # Main fusion loss
        loss_fusion = self.ce_loss(outputs["logits"], labels)
        
        # Auxiliary losses
        loss_audio = 0
        loss_visual = 0
        
        if "audio_features" in outputs:
            audio_pooled = outputs["audio_features"].mean(dim=1)  # [B, 512]
            audio_logits = self.audio_classifier(audio_pooled)
            loss_audio = self.ce_loss(audio_logits, labels)
        
        if "visual_features" in outputs:
            visual_pooled = outputs["visual_features"].mean(dim=1)  # [B, 768]
            visual_logits = self.visual_classifier(visual_pooled)
            loss_visual = self.ce_loss(visual_logits, labels)
        
        # Total loss
        total_loss = (
            self.alpha_fusion * loss_fusion +
            self.alpha_audio * loss_audio +
            self.alpha_visual * loss_visual
        )
        
        return {
            "loss": total_loss,
            "loss_fusion": loss_fusion,
            "loss_audio": loss_audio if isinstance(loss_audio, torch.Tensor) else torch.tensor(0.0),
            "loss_visual": loss_visual if isinstance(loss_visual, torch.Tensor) else torch.tensor(0.0),
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for audio-visual alignment.
    
    Args:
        temperature: Temperature for softmax
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: [B, D] - Audio features
            visual_features: [B, D] - Visual features
            
        Returns:
            loss: Contrastive loss
        """
        # Normalize features
        audio_norm = F.normalize(audio_features, dim=-1)
        visual_norm = F.normalize(visual_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_norm, visual_norm.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = audio_features.size(0)
        labels = torch.arange(batch_size).to(audio_features.device)
        
        # Bidirectional contrastive loss
        loss_a2v = F.cross_entropy(similarity, labels)
        loss_v2a = F.cross_entropy(similarity.T, labels)
        
        loss = (loss_a2v + loss_v2a) / 2
        
        return loss
