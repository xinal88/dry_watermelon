"""
Test script for SigLIP2 integration with Visual Branch.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def test_siglip2_encoder():
    """Test SigLIP2 encoder directly."""
    print("="*60)
    print("Testing SigLIP2 Encoder")
    print("="*60)
    
    from models.visual_branch.siglip_encoder import SigLIPEncoder
    
    # Create encoder
    print("\n1. Loading SigLIP2 encoder...")
    encoder = SigLIPEncoder(
        pretrained_model="google/siglip2-base-patch16-224",
        feature_dim=768,
        freeze_encoder=True,
        backend="transformers",
    )
    
    # Test with dummy input
    print("\n2. Testing with dummy video frames...")
    B, T, C, H, W = 2, 4, 3, 224, 224
    frames = torch.randn(B, T, C, H, W)
    print(f"   Input shape: {frames.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = encoder(frames)
    
    print(f"   Patch tokens: {output['patch_tokens'].shape}")
    print(f"   Pooled features: {output['pooled'].shape}")
    
    # Parameter count
    params = encoder.count_parameters(trainable_only=False)
    trainable = encoder.count_parameters(trainable_only=True)
    print(f"\n3. Parameters:")
    print(f"   Total: {params:,} ({params/1e6:.1f}M)")
    print(f"   Trainable: {trainable:,}")
    
    print("\n[OK] SigLIP2 encoder test passed!")
    return encoder


def test_full_visual_branch():
    """Test complete Visual Branch with SigLIP2."""
    print("\n" + "="*60)
    print("Testing Full Visual Branch with SigLIP2")
    print("="*60)
    
    from models.visual_branch import VisualBranch
    
    # Create Visual Branch
    print("\n1. Creating Visual Branch...")
    visual_branch = VisualBranch(
        pretrained_model="google/siglip2-base-patch16-224",
        feature_dim=768,
        freeze_encoder=True,
        num_keep_tokens=64,
        num_global_tokens=4,
        temporal_depth=4,  # Smaller for testing
        num_segments=8,
    )
    
    # Print summary
    visual_branch.print_summary()
    
    # Test with dummy video
    print("\n2. Testing forward pass...")
    B, T, C, H, W = 1, 8, 3, 224, 224
    video = torch.randn(B, T, C, H, W)
    print(f"   Input video: {video.shape}")
    
    with torch.no_grad():
        output = visual_branch(video, return_intermediates=True)
    
    print(f"   Segment features: {output['segment_features'].shape}")
    print(f"   Frame features: {output['frame_features'].shape}")
    print(f"   Patch tokens: {output['patch_tokens'].shape}")
    print(f"   Compressed tokens: {output['compressed_tokens'].shape}")
    
    print("\n[OK] Full Visual Branch test passed!")
    return visual_branch


def main():
    print("\n" + "#"*60)
    print("# SigLIP2 + Visual Branch Integration Test")
    print("#"*60)
    
    # Test encoder
    test_siglip2_encoder()
    
    # Test full branch
    test_full_visual_branch()
    
    print("\n" + "#"*60)
    print("# All tests passed successfully!")
    print("#"*60)


if __name__ == "__main__":
    main()

