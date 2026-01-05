"""
LFM2-based Multimodal Fusion Module

Uses Liquid LFM2-700M architecture for fusing audio and visual features.
LFM2 combines:
- Short Convolution layers (Lfm2ShortConv) for local patterns
- Attention layers (Lfm2Attention) for global dependencies
- MLP layers for feature transformation

Pipeline:
    Audio Features [B, S, 512] ─┐
                                  ├─> Project → LFM2 Layers → Fused [B, S, D]
    Visual Features [B, S, 768] ─┘
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
import warnings


@dataclass
class LFM2FusionConfig:
    """Configuration for LFM2 Fusion Module."""
    
    # Input dimensions
    audio_dim: int = 512
    visual_dim: int = 768
    num_segments: int = 8
    
    # LFM2 model
    pretrained_model: str = "LiquidAI/LFM2-700M"
    use_pretrained: bool = True  # Load pretrained LFM2-700M
    freeze_backbone: bool = False
    
    # Fusion architecture
    hidden_dim: int = 1536  # LFM2-700M hidden size
    num_layers: int = 6  # Use subset of LFM2 layers
    dropout: float = 0.1
    
    # Projection
    use_gated_projection: bool = True
    projection_hidden_dim: int = 1024
    
    # Output
    output_dim: int = 512  # For classifier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audio_dim": self.audio_dim,
            "visual_dim": self.visual_dim,
            "num_segments": self.num_segments,
            "pretrained_model": self.pretrained_model,
            "use_pretrained": self.use_pretrained,
            "freeze_backbone": self.freeze_backbone,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_gated_projection": self.use_gated_projection,
            "projection_hidden_dim": self.projection_hidden_dim,
            "output_dim": self.output_dim,
        }


class ModalityProjection(nn.Module):
    """
    Project modality-specific features to LFM2 hidden dimension.
    Uses gated projection for better feature integration.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_gated: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.use_gated = use_gated
        
        if use_gated:
            # Gated projection (similar to LFM2's gating mechanism)
            self.gate_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.value_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_proj = nn.Linear(hidden_dim, output_dim)
        else:
            # Simple projection
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D_in] - Input features
        Returns:
            [B, S, D_out] - Projected features
        """
        if self.use_gated:
            gate = torch.sigmoid(self.gate_proj(x))
            value = self.value_proj(x)
            x = self.out_proj(gate * value)
        else:
            x = self.proj(x)
        
        return self.norm(x)


class LFM2Fusion(nn.Module):
    """
    LFM2-based Multimodal Fusion Module.
    
    Fuses audio and visual features using Liquid LFM2 architecture.
    Can use pretrained LFM2-700M weights or train from scratch.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        visual_dim: int = 768,
        num_segments: int = 8,
        pretrained_model: str = "LiquidAI/LFM2-700M",
        use_pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 1536,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_gated_projection: bool = True,
        projection_hidden_dim: int = 1024,
        output_dim: int = 512,
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_pretrained = use_pretrained
        
        # Modality projections
        self.audio_proj = ModalityProjection(
            input_dim=audio_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=hidden_dim,
            use_gated=use_gated_projection,
            dropout=dropout,
        )
        
        self.visual_proj = ModalityProjection(
            input_dim=visual_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=hidden_dim,
            use_gated=use_gated_projection,
            dropout=dropout,
        )
        
        # Modality type embeddings (learnable)
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.visual_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # LFM2 backbone
        if use_pretrained:
            print(f"  Loading pretrained LFM2: {pretrained_model}...")
            self.backbone = self._load_pretrained_lfm2(
                pretrained_model,
                num_layers,
                freeze_backbone,
            )
        else:
            print(f"  Using custom LFM2 implementation (lightweight)...")
            self.backbone = self._create_custom_lfm2(
                hidden_dim,
                num_layers,
                dropout,
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        print(f"\n[OK] LFM2 Fusion initialized:")
        print(f"  - Audio: {audio_dim} -> {hidden_dim}")
        print(f"  - Visual: {visual_dim} -> {hidden_dim}")
        print(f"  - LFM2 layers: {num_layers}")
        print(f"  - Output: {output_dim}")
        print(f"  - Mode: {'Pretrained' if use_pretrained else 'Custom (lightweight)'}")
        if use_pretrained and freeze_backbone:
            print(f"  - Frozen: {freeze_backbone}")
    
    def _load_pretrained_lfm2(
        self,
        model_name: str,
        num_layers: int,
        freeze: bool,
    ) -> nn.Module:
        """Load pretrained LFM2 model and extract layers."""
        try:
            from transformers import AutoModel
            
            print(f"  Loading pretrained LFM2: {model_name}...")
            
            # Load full model
            full_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # Extract decoder layers (the core LFM2 layers)
            if hasattr(full_model, 'model') and hasattr(full_model.model, 'layers'):
                all_layers = full_model.model.layers
            elif hasattr(full_model, 'layers'):
                all_layers = full_model.layers
            else:
                raise AttributeError("Cannot find LFM2 layers in model")
            
            # Use first num_layers
            selected_layers = nn.ModuleList(all_layers[:num_layers])
            
            # Freeze if specified
            if freeze:
                for param in selected_layers.parameters():
                    param.requires_grad = False
                print(f"  [OK] Frozen {num_layers} LFM2 layers")
            else:
                print(f"  [OK] Loaded {num_layers} trainable LFM2 layers")
            
            return selected_layers
            
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained LFM2: {e}\n"
                "Falling back to custom implementation."
            )
            return self._create_custom_lfm2(self.hidden_dim, num_layers, 0.1)
    
    def _create_custom_lfm2(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Create custom LFM2-style layers from scratch."""
        from .lfm2_layers import LFM2DecoderLayer
        
        layers = nn.ModuleList([
            LFM2DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                use_conv=(i % 3 != 2),  # Conv layers except every 3rd
            )
            for i in range(num_layers)
        ])
        
        print(f"  [OK] Created {num_layers} custom LFM2 layers")
        return layers
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal fusion.
        
        Args:
            audio_features: [B, S, D_audio] - Audio segment features
            visual_features: [B, S, D_visual] - Visual segment features
            return_intermediates: Return intermediate features
            
        Returns:
            {
                "fused_features": [B, S, D_out] - Fused features
                "pooled_features": [B, D_out] - Pooled across segments
                "audio_proj": [B, S, D_hidden] - (optional) Projected audio
                "visual_proj": [B, S, D_hidden] - (optional) Projected visual
            }
        """
        B, S, _ = audio_features.shape
        
        # 1. Project modalities to hidden_dim
        audio_proj = self.audio_proj(audio_features)  # [B, S, D_hidden]
        visual_proj = self.visual_proj(visual_features)  # [B, S, D_hidden]
        
        # 2. Add modality type embeddings
        audio_proj = audio_proj + self.audio_type_embed
        visual_proj = visual_proj + self.visual_type_embed
        
        # 3. Concatenate along sequence dimension
        # [B, 2*S, D_hidden] - interleave audio and visual
        fused = torch.stack([audio_proj, visual_proj], dim=2)  # [B, S, 2, D]
        fused = fused.view(B, 2 * S, self.hidden_dim)  # [B, 2*S, D]
        
        # 4. Pass through LFM2 layers
        for layer in self.backbone:
            # LFM2 layers expect specific input format
            if hasattr(layer, 'forward'):
                # For pretrained layers
                fused = layer(fused)[0] if isinstance(layer(fused), tuple) else layer(fused)
            else:
                # For custom layers
                fused = layer(fused)
        
        # 5. Separate back to audio and visual
        fused = fused.view(B, S, 2, self.hidden_dim)
        audio_fused = fused[:, :, 0, :]  # [B, S, D]
        visual_fused = fused[:, :, 1, :]  # [B, S, D]
        
        # 6. Combine (average fusion)
        combined = (audio_fused + visual_fused) / 2  # [B, S, D]
        
        # 7. Output projection
        output = self.output_proj(combined)  # [B, S, D_out]
        
        # 8. Temporal pooling
        pooled = output.mean(dim=1)  # [B, D_out]
        
        result = {
            "fused_features": output,  # [B, S, D_out]
            "pooled_features": pooled,  # [B, D_out]
        }
        
        if return_intermediates:
            result["audio_proj"] = audio_proj
            result["visual_proj"] = visual_proj
            result["audio_fused"] = audio_fused
            result["visual_fused"] = visual_fused
        
        return result
    
    @classmethod
    def from_config(cls, config: LFM2FusionConfig) -> "LFM2Fusion":
        """Create from config object."""
        return cls(**config.to_dict())
    
    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """Count parameters by component."""
        def count(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())
        
        return {
            "audio_proj": count(self.audio_proj),
            "visual_proj": count(self.visual_proj),
            "backbone": count(self.backbone),
            "output_proj": count(self.output_proj),
            "total": count(self),
        }
    
    def print_summary(self):
        """Print model summary."""
        params = self.count_parameters()
        print("\n" + "="*50)
        print("LFM2 Fusion Summary")
        print("="*50)
        print(f"Audio Projection:  {params['audio_proj']:>12,} params")
        print(f"Visual Projection: {params['visual_proj']:>12,} params")
        print(f"LFM2 Backbone:     {params['backbone']:>12,} params")
        print(f"Output Projection: {params['output_proj']:>12,} params")
        print("-"*50)
        print(f"Total:             {params['total']:>12,} params")
        print(f"                   ({params['total']/1e6:.2f}M)")
        print("="*50)
