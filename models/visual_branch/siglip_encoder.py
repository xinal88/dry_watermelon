"""
SigLIP2 Vision Encoder for Video Frames

Extracts patch tokens from video frames using SigLIP2 vision encoder.
Supports HuggingFace transformers backend with AutoModel.

SigLIP2 improvements over SigLIP:
- Decoder loss for better semantic understanding
- Global-local and masked prediction loss
- Aspect ratio and resolution adaptibility

Output: [B, T, N, D] where N = num_patches
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import warnings


class SigLIPEncoder(nn.Module):
    """
    SigLIP2 Vision Encoder wrapper for video frame processing.

    Features:
    - Extract patch tokens from each video frame
    - Support for SigLIP2 pretrained models (google/siglip2-*)
    - Efficient batch processing of video frames
    - Backward compatible with SigLIP1 models
    """

    def __init__(
        self,
        pretrained_model: str = "google/siglip2-base-patch16-224",
        feature_dim: int = 768,
        freeze_encoder: bool = False,
        use_cls_token: bool = True,
        output_hidden_states: bool = False,
        backend: str = "transformers",  # "transformers" or "timm"
        use_pretrained: bool = True,  # Whether to load pretrained weights
    ):
        """
        Args:
            pretrained_model: Pretrained model name/path
            feature_dim: Output feature dimension
            freeze_encoder: Whether to freeze encoder weights
            use_cls_token: Whether to include CLS token in output
            output_hidden_states: Return intermediate hidden states
            backend: "transformers" or "timm"
            use_pretrained: Whether to load pretrained SigLIP (False = custom encoder)
        """
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.feature_dim = feature_dim
        self.freeze_encoder = freeze_encoder
        self.use_cls_token = use_cls_token
        self.output_hidden_states = output_hidden_states
        self.backend = backend
        self.use_pretrained = use_pretrained
        
        # Initialize encoder
        if use_pretrained:
            self._init_encoder()
        else:
            # Use custom encoder directly
            self.backend = "custom"  # Set backend to custom
            self._init_custom_encoder()
        
        # Projection layer if dimensions don't match
        if self.encoder_dim != feature_dim:
            self.projection = nn.Linear(self.encoder_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if specified
        if freeze_encoder and use_pretrained:
            self._freeze_encoder()
    
    def _init_encoder(self):
        """Initialize vision encoder based on backend."""
        if self.backend == "transformers":
            self._init_transformers_encoder()
        elif self.backend == "timm":
            self._init_timm_encoder()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _init_transformers_encoder(self):
        """Initialize using HuggingFace transformers."""
        try:
            from transformers import AutoModel, AutoProcessor
            import torch

            # Load model - AutoModel works for both SigLIP and SigLIP2
            # Use low_cpu_mem_usage to reduce memory footprint
            self.model = AutoModel.from_pretrained(
                self.pretrained_model,
                low_cpu_mem_usage=True,
                dtype=torch.float16,  # Use FP16 to save memory
            )
            self.processor = AutoProcessor.from_pretrained(self.pretrained_model)

            # SigLIP2 uses combined model, extract vision tower
            # For siglip2, model.vision_model contains the vision encoder
            if hasattr(self.model, 'vision_model'):
                self.encoder = self.model.vision_model
                vision_config = self.model.config.vision_config
            else:
                # Fallback for direct vision models
                self.encoder = self.model
                vision_config = self.model.config

            # Get encoder dimension
            self.encoder_dim = vision_config.hidden_size
            self.patch_size = vision_config.patch_size
            self.image_size = vision_config.image_size
            self.num_patches = (self.image_size // self.patch_size) ** 2

            # Detect SigLIP version
            is_siglip2 = "siglip2" in self.pretrained_model.lower()
            version_str = "SigLIP2" if is_siglip2 else "SigLIP"

            print(f"[OK] Loaded {version_str} (transformers): {self.pretrained_model}")
            print(f"  - Hidden size: {self.encoder_dim}")
            print(f"  - Patch size: {self.patch_size}")
            print(f"  - Image size: {self.image_size}")
            print(f"  - Num patches: {self.num_patches}")

        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        except Exception as e:
            warnings.warn(
                f"Failed to load SigLIP model: {e}\n"
                "This might be due to insufficient memory. Falling back to custom implementation."
            )
            # Fallback to custom implementation
            return self._init_custom_encoder()
    
    def _init_custom_encoder(self):
        """Initialize custom lightweight vision encoder as fallback."""
        # Simple CNN-based vision encoder
        class CustomVisionEncoder(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.hidden_size = hidden_size
                
                # Simple CNN backbone
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(256, hidden_size, 3, padding=1),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(),
                )
                
            def forward(self, pixel_values):
                # pixel_values: [B, C, H, W]
                features = self.conv_layers(pixel_values)  # [B, D, H', W']
                B, D, H, W = features.shape
                
                # Reshape to [B, N, D] where N = H*W
                features = features.view(B, D, H * W).transpose(1, 2)  # [B, N, D]
                
                # Create dummy output structure
                class Output:
                    def __init__(self, hidden_states, pooled):
                        self.last_hidden_state = hidden_states
                        self.pooler_output = pooled
                
                pooled = features.mean(dim=1)  # [B, D]
                return Output(features, pooled)
        
        self.encoder = CustomVisionEncoder(hidden_size=self.feature_dim)
        self.encoder_dim = self.feature_dim
        self.patch_size = 16
        self.image_size = 224
        self.num_patches = (224 // 16) ** 2  # 196
        self.processor = None
        self.backend = "custom"
        
        print(f"[OK] Initialized custom vision encoder (lightweight CNN)")
        print(f"  - Hidden size: {self.encoder_dim}")
        print(f"  - Num patches: {self.num_patches}")
        
        return self.encoder
    
    def _init_timm_encoder(self):
        """Initialize using timm library."""
        try:
            import timm
            
            # Map model names
            timm_model_name = self._get_timm_model_name()
            
            self.encoder = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            
            # Get encoder dimension
            self.encoder_dim = self.encoder.num_features
            self.patch_size = self.encoder.patch_embed.patch_size[0]
            self.image_size = self.encoder.patch_embed.img_size[0]
            self.num_patches = (self.image_size // self.patch_size) ** 2
            
            # Store processor as None for timm (use manual normalization)
            self.processor = None
            
            print(f"[OK] Loaded SigLIP (timm): {timm_model_name}")
            print(f"  - Hidden size: {self.encoder_dim}")
            
        except ImportError:
            raise ImportError("timm not installed. Run: pip install timm")
    
    def _get_timm_model_name(self) -> str:
        """Map HuggingFace model name to timm model name."""
        mapping = {
            # SigLIP 1
            "google/siglip-base-patch16-224": "vit_base_patch16_siglip_224",
            "google/siglip-base-patch16-256": "vit_base_patch16_siglip_256",
            "google/siglip-base-patch16-384": "vit_base_patch16_siglip_384",
            "google/siglip-large-patch16-256": "vit_large_patch16_siglip_256",
            "google/siglip-large-patch16-384": "vit_large_patch16_siglip_384",
            "google/siglip-so400m-patch14-384": "vit_so400m_patch14_siglip_384",
            # SigLIP 2 (Note: timm may not have SigLIP2 yet, use transformers backend)
            "google/siglip2-base-patch16-224": "vit_base_patch16_siglip_224",  # Fallback
            "google/siglip2-base-patch16-256": "vit_base_patch16_siglip_256",
            "google/siglip2-large-patch16-256": "vit_large_patch16_siglip_256",
        }
        if self.pretrained_model not in mapping:
            warnings.warn(
                f"Model {self.pretrained_model} not in timm mapping. "
                "Consider using backend='transformers' for SigLIP2 models."
            )
        return mapping.get(self.pretrained_model, "vit_base_patch16_siglip_224")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[OK] Encoder weights frozen")

    def forward(
        self,
        frames: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for video frames.

        Args:
            frames: Video frames tensor
                - Shape [B, T, C, H, W] for video
                - Shape [B, C, H, W] for single images
            return_dict: Whether to return dict or just features

        Returns:
            If return_dict=True:
                {
                    "patch_tokens": [B, T, N, D] - Patch tokens per frame
                    "cls_tokens": [B, T, D] - CLS token per frame (if use_cls_token)
                    "pooled": [B, T, D] - Pooled features per frame
                }
            Else:
                patch_tokens: [B, T, N, D]
        """
        # Handle input shape
        is_video = frames.dim() == 5
        if not is_video:
            frames = frames.unsqueeze(1)  # [B, 1, C, H, W]

        B, T, C, H, W = frames.shape

        # Reshape for batch processing: [B*T, C, H, W]
        frames_flat = frames.view(B * T, C, H, W)

        # Encode frames
        if self.backend == "transformers":
            outputs = self.encoder(
                pixel_values=frames_flat,
                output_hidden_states=self.output_hidden_states,
            )

            # SigLIP returns last_hidden_state: [B*T, N+1, D]
            # First token is CLS, rest are patch tokens
            hidden_states = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # [B*T, D]

        elif self.backend == "timm":
            # timm forward_features returns [B*T, N+1, D]
            hidden_states = self.encoder.forward_features(frames_flat)
            pooled_output = hidden_states[:, 0]  # CLS token as pooled
        
        elif self.backend == "custom":
            # Custom encoder
            outputs = self.encoder(frames_flat)
            hidden_states = outputs.last_hidden_state  # [B*T, N, D]
            pooled_output = outputs.pooler_output  # [B*T, D]
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Project to feature_dim
        hidden_states = self.projection(hidden_states)  # [B*T, N+1, D] or [B*T, N, D]
        pooled_output = self.projection(pooled_output) if hasattr(self, 'projection') else pooled_output

        # Split CLS and patch tokens
        if self.backend == "custom":
            # Custom encoder doesn't have CLS token
            patch_tokens = hidden_states  # [B*T, N, D]
            cls_tokens = pooled_output.unsqueeze(1)  # [B*T, 1, D] - use pooled as CLS
        else:
            # Transformers/timm have CLS token
            cls_tokens = hidden_states[:, 0]  # [B*T, D]
            patch_tokens = hidden_states[:, 1:]  # [B*T, N, D]

        # Reshape back to video format
        N = patch_tokens.shape[1]
        D = patch_tokens.shape[2]

        patch_tokens = patch_tokens.view(B, T, N, D)  # [B, T, N, D]
        cls_tokens = cls_tokens.view(B, T, D)  # [B, T, D]
        pooled_output = pooled_output.view(B, T, D)  # [B, T, D]

        if not return_dict:
            return patch_tokens

        result = {
            "patch_tokens": patch_tokens,  # [B, T, N, D]
            "pooled": pooled_output,  # [B, T, D]
        }

        if self.use_cls_token:
            result["cls_tokens"] = cls_tokens  # [B, T, D]

        return result

    def get_num_patches(self) -> int:
        """Return number of patches per frame."""
        return self.num_patches

    def get_patch_grid_size(self) -> Tuple[int, int]:
        """Return patch grid dimensions (H_patches, W_patches)."""
        grid_size = self.image_size // self.patch_size
        return (grid_size, grid_size)

    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self.feature_dim

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

