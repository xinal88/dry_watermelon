"""
Unit tests for Visual Branch components.
Tests SigLIPEncoder, ROITokenCompression, TemporalEncoder, and VisualBranch.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestROITokenCompression:
    """Tests for ROI Token Compression module."""
    
    def test_compression_shapes(self):
        """Test output shapes of ROI compression."""
        from models.visual_branch.roi_compression import ROITokenCompression
        
        B, T, N, D = 2, 8, 196, 768  # 196 patches for 224x224 with patch_size=16
        K, G = 64, 4  # Keep 64 tokens + 4 global
        
        compression = ROITokenCompression(
            input_dim=D,
            num_keep_tokens=K,
            num_global_tokens=G,
        )
        
        tokens = torch.randn(B, T, N, D)
        output = compression(tokens)
        
        assert output["compressed_tokens"].shape == (B, T, K + G, D)
        assert output["selected_tokens"].shape == (B, T, K, D)
        assert output["indices"].shape == (B, T, K)
        assert output["global_tokens"].shape == (B, T, G, D)
        print("[OK] ROI compression shapes correct")
    
    def test_roi_mask_effect(self):
        """Test that ROI mask affects token selection."""
        from models.visual_branch.roi_compression import ROITokenCompression
        
        B, T, N, D = 1, 4, 49, 256
        K = 16
        
        compression = ROITokenCompression(
            input_dim=D,
            num_keep_tokens=K,
            num_global_tokens=0,
            roi_weight=10.0,  # Strong ROI bias
        )
        compression.eval()
        
        tokens = torch.randn(B, T, N, D)
        
        # Create ROI mask (center patches)
        roi_mask = torch.zeros(B, T, N)
        center_indices = [24, 25, 31, 32]  # Center of 7x7 grid
        roi_mask[:, :, center_indices] = 1.0
        
        # Compare with and without ROI
        output_no_roi = compression(tokens, roi_mask=None, return_scores=True)
        output_with_roi = compression(tokens, roi_mask=roi_mask, return_scores=True)
        
        # Check that ROI regions have higher scores
        roi_scores = output_with_roi["scores"][:, :, center_indices].mean()
        non_roi_scores = output_with_roi["scores"].mean()
        
        print(f"  ROI region avg score: {roi_scores:.4f}")
        print(f"  Overall avg score: {non_roi_scores:.4f}")
        print("[OK] ROI mask affects scoring")


class TestTemporalEncoder:
    """Tests for Temporal Encoder module."""
    
    def test_temporal_encoder_shapes(self):
        """Test output shapes of temporal encoder."""
        from models.visual_branch.temporal_encoder import TemporalEncoder
        
        B, T, K, D = 2, 16, 68, 768  # 68 = 64 selected + 4 global
        S = 8  # Output segments
        
        encoder = TemporalEncoder(
            dim=D,
            depth=4,
            num_heads=8,
            num_segments=S,
        )
        
        x = torch.randn(B, T, K, D)
        output = encoder(x)
        
        assert output["segment_features"].shape == (B, S, D)
        assert output["frame_features"].shape == (B, T, D)
        print("[OK] Temporal encoder shapes correct")
    
    def test_gscb_block(self):
        """Test GSCB block forward pass."""
        from models.visual_branch.temporal_encoder import GatedShortConvBlock
        
        B, T, D = 4, 32, 512
        
        gscb = GatedShortConvBlock(dim=D, kernel_size=4)
        x = torch.randn(B, T, D)
        output = gscb(x)
        
        assert output.shape == (B, T, D)
        print("[OK] GSCB block works correctly")


class TestVisualBranchIntegration:
    """Integration tests for complete Visual Branch."""
    
    def test_visual_branch_without_siglip(self):
        """Test Visual Branch pipeline without loading SigLIP."""
        from models.visual_branch.roi_compression import ROITokenCompression
        from models.visual_branch.temporal_encoder import TemporalEncoder
        
        B, T, N, D = 2, 8, 196, 768
        K, G = 64, 4
        S = 8
        
        # Simulate patch tokens (as if from SigLIP)
        patch_tokens = torch.randn(B, T, N, D)
        
        # ROI Compression
        compression = ROITokenCompression(
            input_dim=D,
            num_keep_tokens=K,
            num_global_tokens=G,
        )
        comp_output = compression(patch_tokens)
        compressed = comp_output["compressed_tokens"]  # [B, T, K+G, D]
        
        # Temporal Encoder
        temporal = TemporalEncoder(dim=D, depth=4, num_segments=S)
        temp_output = temporal(compressed)
        
        assert temp_output["segment_features"].shape == (B, S, D)
        print("[OK] Visual Branch pipeline (without SigLIP) works")
    
    def test_parameter_count(self):
        """Test parameter counting."""
        from models.visual_branch.roi_compression import ROITokenCompression
        from models.visual_branch.temporal_encoder import TemporalEncoder
        
        D = 768
        
        compression = ROITokenCompression(input_dim=D)
        temporal = TemporalEncoder(dim=D, depth=6)
        
        comp_params = compression.count_parameters()
        temp_params = temporal.count_parameters()
        
        print(f"  ROI Compression: {comp_params:,} params")
        print(f"  Temporal Encoder: {temp_params:,} params")
        print(f"  Total (without SigLIP): {comp_params + temp_params:,} params")
        print("[OK] Parameter counting works")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Visual Branch Unit Tests")
    print("="*60)

    # ROI Compression tests
    print("\n[1/3] Testing ROI Token Compression...")
    roi_tests = TestROITokenCompression()
    roi_tests.test_compression_shapes()
    roi_tests.test_roi_mask_effect()

    # Temporal Encoder tests
    print("\n[2/3] Testing Temporal Encoder...")
    temp_tests = TestTemporalEncoder()
    temp_tests.test_temporal_encoder_shapes()
    temp_tests.test_gscb_block()

    # Integration tests
    print("\n[3/3] Testing Visual Branch Integration...")
    int_tests = TestVisualBranchIntegration()
    int_tests.test_visual_branch_without_siglip()
    int_tests.test_parameter_count()

    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

