"""
FastConformer Encoder for Audio Processing
Wrapper around NVIDIA NeMo's FastConformer or HuggingFace implementation.

FastConformer combines:
- Efficient Conformer architecture
- Fast attention mechanisms
- Optimized for speech/audio tasks
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings


class FastConformerEncoder(nn.Module):
    """
    FastConformer encoder wrapper with flexible backend support.
    
    Supports:
    1. NVIDIA NeMo FastConformer (recommended)
    2. HuggingFace Transformers (fallback)
    3. Custom lightweight implementation
    """
    
    def __init__(
        self,
        pretrained_model: Optional[str] = None,
        feature_dim: int = 512,
        freeze_encoder: bool = False,
        use_nemo: bool = True,
        # Audio preprocessing params
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        # Model params
        num_layers: int = 17,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.freeze_encoder = freeze_encoder
        self.sample_rate = sample_rate
        
        # Audio preprocessing
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Initialize encoder based on backend
        if use_nemo and pretrained_model:
            self.encoder = self._init_nemo_encoder(pretrained_model)
            self.backend = "nemo"
        elif pretrained_model and "facebook" in pretrained_model.lower():
            # HuggingFace Wav2Vec2 or similar
            self.encoder = self._init_hf_encoder(pretrained_model)
            self.backend = "huggingface"
        else:
            # Custom lightweight implementation
            self.encoder = self._init_custom_encoder(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.backend = "custom"
        
        # Projection layer to match feature_dim
        encoder_dim = self._get_encoder_dim()
        if encoder_dim != feature_dim:
            self.projection = nn.Linear(encoder_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
    
    def _init_nemo_encoder(self, pretrained_model: str):
        """Initialize NVIDIA NeMo FastConformer"""
        try:
            from nemo.collections.asr.models import EncDecCTCModel
            
            # Load pretrained model
            model = EncDecCTCModel.from_pretrained(pretrained_model)
            
            # Extract encoder only
            encoder = model.encoder
            
            print(f"[OK] Loaded NeMo FastConformer: {pretrained_model}")
            return encoder
            
        except ImportError:
            warnings.warn(
                "NeMo not installed. Install with: pip install nemo_toolkit[asr]\n"
                "Falling back to custom implementation."
            )
            return self._init_custom_encoder()
        except Exception as e:
            warnings.warn(f"Failed to load NeMo model: {e}\nFalling back to custom implementation.")
            return self._init_custom_encoder()
    
    def _init_hf_encoder(self, pretrained_model: str):
        """Initialize HuggingFace encoder (e.g., Wav2Vec2)"""
        try:
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(pretrained_model)
            print(f"[OK] Loaded HuggingFace model: {pretrained_model}")
            return model
            
        except Exception as e:
            warnings.warn(f"Failed to load HuggingFace model: {e}\nFalling back to custom implementation.")
            return self._init_custom_encoder()
    
    def _init_custom_encoder(
        self,
        num_layers: int = 17,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize custom lightweight Conformer encoder"""
        from .conformer_blocks import ConformerEncoder
        
        encoder = ConformerEncoder(
            input_dim=80,  # n_mels
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        print(f"[OK] Initialized custom Conformer encoder ({num_layers} layers, {d_model} dim)")
        return encoder
    
    def _get_encoder_dim(self) -> int:
        """Get output dimension of encoder"""
        if self.backend == "nemo":
            return self.encoder.d_model
        elif self.backend == "huggingface":
            return self.encoder.config.hidden_size
        elif self.backend == "custom":
            return self.encoder.d_model
        else:
            return 512  # default
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[OK] Encoder frozen")
    
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FastConformer encoder.
        
        Args:
            audio: [B, T_audio] - Raw audio waveform or [B, T, n_mels] - Mel spectrogram
            audio_lengths: [B] - Actual lengths of audio sequences
        
        Returns:
            Dictionary containing:
                - features: [B, T, D] - Encoded features
                - lengths: [B] - Output sequence lengths
        """
        # Preprocess audio if raw waveform
        if audio.dim() == 2:  # [B, T_audio]
            audio = self.preprocessor(audio)  # [B, T, n_mels]
        
        # Encode based on backend
        if self.backend == "nemo":
            encoded, encoded_lengths = self.encoder(audio_signal=audio, length=audio_lengths)
        elif self.backend == "huggingface":
            outputs = self.encoder(audio)
            encoded = outputs.last_hidden_state
            encoded_lengths = audio_lengths
        else:  # custom
            encoded, encoded_lengths = self.encoder(audio, audio_lengths)
        
        # Project to feature_dim
        features = self.projection(encoded)  # [B, T, feature_dim]
        
        return {
            "features": features,
            "lengths": encoded_lengths
        }


class AudioPreprocessor(nn.Module):
    """Audio preprocessing: waveform -> mel spectrogram"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Use torchaudio for mel spectrogram
        import torchaudio.transforms as T
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.
        
        Args:
            waveform: [B, T] - Audio waveform
        Returns:
            mel_spec: [B, T', n_mels] - Mel spectrogram
        """
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [B, n_mels, T']
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Transpose to [B, T', n_mels]
        mel_spec = mel_spec.transpose(1, 2)
        
        return mel_spec

