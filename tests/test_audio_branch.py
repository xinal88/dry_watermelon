"""
Unit tests for Audio Branch
Tests FastConformer + Segment Attention Pooling
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.audio_branch import (
    AudioBranch,
    AudioBranchConfig,
    SegmentAttentionPooling,
    FastConformerEncoder,
    AttentionPooling
)


class TestSegmentAttentionPooling:
    """Test Segment Attention Pooling module"""
    
    def test_segment_pooling_shape(self):
        """Test output shape of segment pooling"""
        batch_size = 4
        seq_len = 256
        dim = 512
        num_segments = 8
        
        # Create module
        pooling = SegmentAttentionPooling(
            dim=dim,
            num_segments=num_segments,
            pooling_type="attention"
        )
        
        # Create input
        x = torch.randn(batch_size, seq_len, dim)
        
        # Forward pass
        output = pooling(x)
        
        # Check shape
        assert output.shape == (batch_size, num_segments, dim)
        print(f"✓ Segment pooling output shape: {output.shape}")
    
    def test_different_pooling_types(self):
        """Test different pooling strategies"""
        batch_size = 2
        seq_len = 128
        dim = 256
        num_segments = 4
        
        x = torch.randn(batch_size, seq_len, dim)
        
        for pooling_type in ["attention", "max", "avg", "learnable"]:
            pooling = SegmentAttentionPooling(
                dim=dim,
                num_segments=num_segments,
                pooling_type=pooling_type
            )
            
            output = pooling(x)
            assert output.shape == (batch_size, num_segments, dim)
            print(f"✓ {pooling_type} pooling works correctly")


class TestFastConformerEncoder:
    """Test FastConformer Encoder"""
    
    def test_custom_conformer(self):
        """Test custom Conformer implementation"""
        batch_size = 2
        seq_len = 100
        n_mels = 80
        feature_dim = 512
        
        # Create encoder (custom implementation)
        encoder = FastConformerEncoder(
            pretrained_model=None,  # Use custom implementation
            feature_dim=feature_dim,
            use_nemo=False,
            num_layers=4,  # Smaller for testing
            d_model=512,
            num_heads=8
        )
        
        # Create input (mel spectrogram)
        mel_spec = torch.randn(batch_size, seq_len, n_mels)
        
        # Forward pass
        output = encoder(mel_spec)
        
        # Check output
        assert "features" in output
        assert output["features"].shape == (batch_size, seq_len, feature_dim)
        print(f"✓ Custom Conformer encoder output shape: {output['features'].shape}")
    
    def test_audio_preprocessing(self):
        """Test audio waveform to mel spectrogram conversion"""
        batch_size = 2
        audio_length = 16000 * 3  # 3 seconds at 16kHz
        feature_dim = 512
        
        # Create encoder
        encoder = FastConformerEncoder(
            pretrained_model=None,
            feature_dim=feature_dim,
            use_nemo=False,
            num_layers=2,
            sample_rate=16000,
            n_mels=80
        )
        
        # Create raw audio waveform
        audio = torch.randn(batch_size, audio_length)
        
        # Forward pass
        output = encoder(audio)
        
        # Check output
        assert "features" in output
        assert output["features"].dim() == 3
        assert output["features"].shape[-1] == feature_dim
        print(f"✓ Audio preprocessing works: {audio.shape} -> {output['features'].shape}")


class TestAudioBranch:
    """Test complete Audio Branch"""
    
    def test_audio_branch_forward(self):
        """Test full audio branch forward pass"""
        batch_size = 4
        audio_length = 16000 * 3  # 3 seconds
        feature_dim = 512
        num_segments = 8
        
        # Create audio branch
        audio_branch = AudioBranch(
            pretrained_model=None,
            feature_dim=feature_dim,
            use_nemo=False,
            num_layers=4,  # Smaller for testing
            num_segments=num_segments,
            pooling_type="attention"
        )
        
        # Create input
        audio = torch.randn(batch_size, audio_length)
        
        # Forward pass
        output = audio_branch(audio, return_all_segments=True)
        
        # Check outputs
        assert "segment_features" in output
        assert output["segment_features"].shape == (batch_size, num_segments, feature_dim)
        print(f"✓ Audio branch output shape: {output['segment_features'].shape}")
        
        # Test single representation mode
        output_single = audio_branch(audio, return_all_segments=False)
        assert output_single["segment_features"].shape == (batch_size, feature_dim)
        print(f"✓ Audio branch single representation: {output_single['segment_features'].shape}")
    
    def test_audio_branch_params(self):
        """Test parameter counting"""
        audio_branch = AudioBranch(
            pretrained_model=None,
            feature_dim=512,
            use_nemo=False,
            num_layers=4,
            num_segments=8
        )
        
        params = audio_branch.get_num_params()
        
        print(f"\n📊 Audio Branch Parameters:")
        print(f"  - Encoder: {params['encoder']:,}")
        print(f"  - Pooling: {params['pooling']:,}")
        print(f"  - Total: {params['total']:,}")
        print(f"  - Trainable: {params['trainable']:,}")
        
        assert params['total'] > 0
        assert params['trainable'] > 0


class TestAudioBranchConfig:
    """Test configuration class"""
    
    def test_config_creation(self):
        """Test config creation and conversion"""
        config = AudioBranchConfig(
            feature_dim=512,
            num_segments=8,
            pooling_type="attention"
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        assert config_dict["feature_dim"] == 512
        assert config_dict["num_segments"] == 8
        
        # Recreate from dict
        new_config = AudioBranchConfig.from_dict(config_dict)
        assert new_config.feature_dim == 512
        assert new_config.num_segments == 8
        
        print("✓ Config creation and conversion works")


def run_all_tests():
    """Run all tests manually"""
    print("=" * 60)
    print("Testing Audio Branch Components")
    print("=" * 60)
    
    # Test Segment Pooling
    print("\n[1/5] Testing Segment Attention Pooling...")
    test_pooling = TestSegmentAttentionPooling()
    test_pooling.test_segment_pooling_shape()
    test_pooling.test_different_pooling_types()
    
    # Test FastConformer
    print("\n[2/5] Testing FastConformer Encoder...")
    test_encoder = TestFastConformerEncoder()
    test_encoder.test_custom_conformer()
    test_encoder.test_audio_preprocessing()
    
    # Test Audio Branch
    print("\n[3/5] Testing Complete Audio Branch...")
    test_branch = TestAudioBranch()
    test_branch.test_audio_branch_forward()
    test_branch.test_audio_branch_params()
    
    # Test Config
    print("\n[4/5] Testing Configuration...")
    test_config = TestAudioBranchConfig()
    test_config.test_config_creation()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

