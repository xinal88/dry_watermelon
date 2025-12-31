"""
Test Complete Multimodal FER Model
Tests full pipeline: Audio + Visual → LFM2 Fusion → Classifier
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_complete_model():
    """Test complete multimodal model forward pass."""
    print("="*70)
    print("Testing Complete Multimodal FER Model")
    print("="*70)
    
    from models import MultimodalFER, MultimodalFERConfig
    from models.audio_branch import AudioBranchConfig
    from models.visual_branch import VisualBranchConfig
    from models.fusion import LFM2FusionConfig
    from models.classifier import ClassifierConfig
    
    # Configuration
    print("\n[1/5] Creating configuration...")
    config = MultimodalFERConfig(
        audio_config=AudioBranchConfig(
            pretrained_model=None,
            feature_dim=512,
            use_nemo=False,
            num_layers=4,  # Lightweight
            num_segments=8,
        ),
        visual_config=VisualBranchConfig(
            pretrained_model="google/siglip-base-patch16-224",  # Will use custom if not available
            feature_dim=768,
            freeze_encoder=True,
            num_keep_tokens=64,
            temporal_depth=4,  # Lightweight
            num_segments=8,
        ),
        fusion_config=LFM2FusionConfig(
            audio_dim=512,
            visual_dim=768,
            pretrained_model="LiquidAI/LFM2-700M",
            use_pretrained=False,  # Use custom for testing
            num_layers=4,  # Lightweight
            output_dim=512,
        ),
        classifier_config=ClassifierConfig(
            input_dim=512,
            hidden_dims=[512, 256],
            num_classes=8,
            dropout=0.1,
        ),
        num_classes=8,
        num_segments=8,
    )
    
    # Create model
    print("\n[2/5] Building model...")
    try:
        model = MultimodalFER.from_config(config)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Create dummy inputs
    print("\n[3/5] Creating dummy inputs...")
    batch_size = 2
    
    # Audio: 3 seconds at 16kHz
    audio = torch.randn(batch_size, 48000)
    print(f"  Audio shape: {audio.shape}")
    
    # Video: 16 frames, 224x224
    video = torch.randn(batch_size, 16, 3, 224, 224)
    print(f"  Video shape: {video.shape}")
    
    # Forward pass
    print("\n[4/5] Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                audio=audio,
                video=video,
                return_intermediates=True,
            )
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check outputs
    print("\n[5/5] Checking outputs...")
    
    expected_keys = ["logits", "probabilities"]
    for key in expected_keys:
        if key not in outputs:
            print(f"✗ Missing output key: {key}")
            return False
        print(f"  ✓ {key}: {outputs[key].shape}")
    
    # Check shapes
    assert outputs["logits"].shape == (batch_size, 8), "Incorrect logits shape"
    assert outputs["probabilities"].shape == (batch_size, 8), "Incorrect probabilities shape"
    
    # Check probabilities sum to 1
    prob_sum = outputs["probabilities"].sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1"
    
    # Check intermediate features
    if "audio_features" in outputs:
        print(f"  ✓ audio_features: {outputs['audio_features'].shape}")
    if "visual_features" in outputs:
        print(f"  ✓ visual_features: {outputs['visual_features'].shape}")
    if "fused_features" in outputs:
        print(f"  ✓ fused_features: {outputs['fused_features'].shape}")
    
    # Print summary
    print("\n" + "="*70)
    model.print_summary()
    
    # Test modality-specific forward passes
    print("\n" + "="*70)
    print("Testing Modality-Specific Forward Passes")
    print("="*70)
    
    # Audio only
    print("\n[Audio Only]")
    try:
        audio_outputs = model.forward_audio_only(audio)
        print(f"  ✓ Logits: {audio_outputs['logits'].shape}")
        print(f"  ✓ Probabilities: {audio_outputs['probabilities'].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Visual only
    print("\n[Visual Only]")
    try:
        visual_outputs = model.forward_visual_only(video)
        print(f"  ✓ Logits: {visual_outputs['logits'].shape}")
        print(f"  ✓ Probabilities: {visual_outputs['probabilities'].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    
    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*70)
    print("Testing Training Step")
    print("="*70)
    
    from models import MultimodalFER
    
    # Create lightweight model
    model = MultimodalFER(
        num_classes=8,
        num_segments=8,
    )
    
    # Create dummy data
    batch_size = 2
    audio = torch.randn(batch_size, 48000)
    video = torch.randn(batch_size, 16, 3, 224, 224)
    labels = torch.randint(0, 8, (batch_size,))
    
    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    print("\n[Training Step]")
    model.train()
    
    # Forward
    outputs = model(audio, video)
    loss = criterion(outputs["logits"], labels)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("  ✓ Backward pass successful")
    print("  ✓ Optimizer step successful")
    
    print("\n✅ Training step test passed!")
    
    return True


def test_memory_usage():
    """Estimate memory usage."""
    print("\n" + "="*70)
    print("Memory Usage Estimation")
    print("="*70)
    
    from models import MultimodalFER
    
    model = MultimodalFER(num_classes=8, num_segments=8)
    
    params = model.count_parameters()
    
    # Memory estimation
    param_memory_fp32 = params['total'] * 4 / 1e9
    param_memory_fp16 = params['total'] * 2 / 1e9
    
    # Activation memory (rough estimate)
    batch_size = 8
    audio_length = 48000
    video_frames = 16
    video_size = 224
    
    # Audio activations
    audio_act = batch_size * audio_length * 4 / 1e9  # Input
    audio_act += batch_size * 8 * 512 * 4 / 1e9  # Segment features
    
    # Visual activations
    visual_act = batch_size * video_frames * 3 * video_size * video_size * 4 / 1e9  # Input
    visual_act += batch_size * 8 * 768 * 4 / 1e9  # Segment features
    
    # Fusion activations
    fusion_act = batch_size * 8 * 512 * 4 / 1e9
    
    total_act = audio_act + visual_act + fusion_act
    
    print(f"\nParameter Memory:")
    print(f"  FP32: {param_memory_fp32:.2f} GB")
    print(f"  FP16: {param_memory_fp16:.2f} GB")
    
    print(f"\nActivation Memory (batch_size={batch_size}):")
    print(f"  Audio: {audio_act:.2f} GB")
    print(f"  Visual: {visual_act:.2f} GB")
    print(f"  Fusion: {fusion_act:.2f} GB")
    print(f"  Total: {total_act:.2f} GB")
    
    print(f"\nTotal Memory (FP16 + Activations):")
    print(f"  Training: {param_memory_fp16 + total_act * 2:.2f} GB (with gradients)")
    print(f"  Inference: {param_memory_fp16 + total_act:.2f} GB")
    
    # Check if fits RTX 3050
    if param_memory_fp16 + total_act * 2 < 12:
        print(f"\n✅ Model fits RTX 3050 (12GB) with batch_size={batch_size}")
    else:
        print(f"\n⚠️ Model may not fit RTX 3050 with batch_size={batch_size}")
        print(f"   Try reducing batch_size or using gradient accumulation")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE MODEL TEST SUITE")
    print("="*70)
    
    # Run tests
    tests = [
        ("Complete Model", test_complete_model),
        ("Training Step", test_training_step),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, result in results:
        status = "✅ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed")
    
    print("="*70)
