"""
Demo script for Visual Branch
Tests the complete pipeline without requiring SigLIP model download.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def demo_visual_branch_pipeline():
    """Demo the visual branch pipeline with synthetic data."""
    print("="*60)
    print("Visual Branch Demo")
    print("="*60)
    
    from models.visual_branch.roi_compression import ROITokenCompression
    from models.visual_branch.temporal_encoder import TemporalEncoder
    
    # Configuration
    B = 2  # Batch size
    T = 16  # Number of frames
    N = 196  # Number of patches (14x14 for 224px with patch_size=16)
    D = 768  # Feature dimension
    K = 64  # Keep tokens
    G = 4  # Global tokens
    S = 8  # Output segments
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Frames: {T}")
    print(f"  Patches per frame: {N}")
    print(f"  Feature dim: {D}")
    print(f"  Keep tokens: {K} + {G} global = {K+G}")
    print(f"  Output segments: {S}")
    
    # Simulate SigLIP output (patch tokens)
    print("\n1. Simulating SigLIP output...")
    patch_tokens = torch.randn(B, T, N, D)
    print(f"   Patch tokens shape: {patch_tokens.shape}")
    
    # Create ROI mask (center region for face)
    print("\n2. Creating ROI mask (center face region)...")
    roi_mask = torch.zeros(B, T, N)
    # Assume 14x14 grid, mark center 6x6 as face region
    grid_size = 14
    for i in range(4, 10):
        for j in range(4, 10):
            roi_mask[:, :, i * grid_size + j] = 1.0
    face_ratio = roi_mask.sum() / roi_mask.numel() * 100
    print(f"   ROI mask shape: {roi_mask.shape}")
    print(f"   Face region: {face_ratio:.1f}% of patches")
    
    # ROI Token Compression
    print("\n3. Applying ROI Token Compression...")
    compression = ROITokenCompression(
        input_dim=D,
        num_keep_tokens=K,
        num_global_tokens=G,
        roi_weight=3.0,
    )
    
    comp_output = compression(patch_tokens, roi_mask=roi_mask, return_scores=True)
    compressed = comp_output["compressed_tokens"]
    scores = comp_output["scores"]
    
    print(f"   Compressed tokens: {compressed.shape}")
    print(f"   Importance scores: {scores.shape}")
    
    # Analyze token selection
    roi_indices = roi_mask[0, 0].nonzero().squeeze()
    roi_scores = scores[0, 0, roi_indices].mean().item()
    non_roi_scores = scores[0, 0, ~roi_mask[0, 0].bool()].mean().item()
    print(f"   Avg ROI score: {roi_scores:.4f}")
    print(f"   Avg non-ROI score: {non_roi_scores:.4f}")
    print(f"   ROI bias ratio: {roi_scores/non_roi_scores:.2f}x")
    
    # Temporal Encoding
    print("\n4. Applying Temporal Encoder...")
    temporal = TemporalEncoder(
        dim=D,
        depth=6,
        num_heads=8,
        gscb_ratio=0.7,
        num_segments=S,
    )
    
    temp_output = temporal(compressed)
    segment_features = temp_output["segment_features"]
    frame_features = temp_output["frame_features"]
    
    print(f"   Segment features: {segment_features.shape}")
    print(f"   Frame features: {frame_features.shape}")
    
    # Parameter counts
    print("\n5. Model Statistics:")
    comp_params = compression.count_parameters()
    temp_params = temporal.count_parameters()
    
    print(f"   ROI Compression: {comp_params:,} params ({comp_params/1e6:.2f}M)")
    print(f"   Temporal Encoder: {temp_params:,} params ({temp_params/1e6:.2f}M)")
    print(f"   Total (without SigLIP): {(comp_params + temp_params):,} params")
    
    # Memory estimation
    print("\n6. Memory Estimation (FP32):")
    params_bytes = (comp_params + temp_params) * 4
    # Activation memory estimate: batch * frames * tokens * dim * 4 bytes * 2 (for grads)
    activation_bytes = B * T * (K + G) * D * 4 * 2
    total_memory = (params_bytes + activation_bytes) / 1e9
    
    print(f"   Parameter memory: {params_bytes/1e6:.1f} MB")
    print(f"   Activation memory (est.): {activation_bytes/1e6:.1f} MB")
    print(f"   Total memory (est.): {total_memory:.2f} GB")
    
    print("\n" + "="*60)
    print("✅ Visual Branch Demo Complete!")
    print("="*60)
    
    return {
        "segment_features": segment_features,
        "frame_features": frame_features,
        "compressed_tokens": compressed,
    }


def demo_with_siglip():
    """Demo with actual SigLIP model (requires model download)."""
    print("\n" + "="*60)
    print("Testing with SigLIP (requires model download)")
    print("="*60)
    
    try:
        from models.visual_branch import VisualBranch
        
        # Create full visual branch
        visual_branch = VisualBranch(
            pretrained_model="google/siglip-base-patch16-224",
            feature_dim=768,
            freeze_encoder=True,
            num_keep_tokens=64,
            num_global_tokens=4,
            temporal_depth=4,
            num_segments=8,
        )
        
        visual_branch.print_summary()
        
        # Test with dummy video
        B, T, C, H, W = 1, 8, 3, 224, 224
        video = torch.randn(B, T, C, H, W)
        
        print(f"\nInput video: {video.shape}")
        output = visual_branch(video)
        print(f"Output segment features: {output['segment_features'].shape}")
        
        print("\n✅ Full Visual Branch with SigLIP works!")
        
    except Exception as e:
        print(f"\n⚠️ SigLIP test skipped: {e}")
        print("   Install transformers and download model to test full pipeline.")


if __name__ == "__main__":
    # Run basic demo (no model download required)
    demo_visual_branch_pipeline()
    
    # Optionally test with SigLIP
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-siglip", action="store_true", help="Test with SigLIP")
    args = parser.parse_args()
    
    if args.with_siglip:
        demo_with_siglip()

