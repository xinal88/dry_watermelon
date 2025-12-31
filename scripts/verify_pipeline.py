"""
Comprehensive Pipeline Verification

Verify all components work correctly before training.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check Python environment."""
    print("\n" + "="*70)
    print("[1/6] Checking Environment")
    print("="*70)
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Running on CPU")
    
    return True


def check_imports():
    """Check all required imports."""
    print("\n" + "="*70)
    print("[2/6] Checking Imports")
    print("="*70)
    
    try:
        from models import MultimodalFER
        print("✓ MultimodalFER")
        
        from training.losses import EmotionLoss
        print("✓ EmotionLoss")
        
        from training.metrics import EmotionMetrics
        print("✓ EmotionMetrics")
        
        from data.test_dataset import create_test_dataloader
        print("✓ Test Dataset")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def check_dataset():
    """Check dataset loading."""
    print("\n" + "="*70)
    print("[3/6] Checking Dataset")
    print("="*70)
    
    try:
        from data.test_dataset import create_test_dataloader
        
        dataloader = create_test_dataloader(batch_size=1)
        print(f"✓ Dataset created: {len(dataloader.dataset)} samples")
        
        # Load one batch
        for audio, video, labels, metadata in dataloader:
            print(f"✓ Batch loaded:")
            print(f"  Audio: {audio.shape}")
            print(f"  Video: {video.shape}")
            print(f"  Labels: {labels}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Dataset check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model():
    """Check model creation."""
    print("\n" + "="*70)
    print("[4/6] Checking Model Creation")
    print("="*70)
    
    try:
        from models import MultimodalFER
        
        print("Creating model...")
        model = MultimodalFER(num_classes=8, num_segments=8)
        print("✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def check_forward_pass(model):
    """Check model forward pass."""
    print("\n" + "="*70)
    print("[5/6] Checking Forward Pass")
    print("="*70)
    
    try:
        # Create dummy inputs
        batch_size = 2
        audio = torch.randn(batch_size, 48000)
        video = torch.randn(batch_size, 16, 3, 224, 224)
        
        print("Running forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(audio, video)
        
        print(f"✓ Forward pass successful")
        print(f"  Logits: {outputs['logits'].shape}")
        print(f"  Probabilities: {outputs['probabilities'].shape}")
        
        # Check probabilities sum to 1
        prob_sum = outputs['probabilities'].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-5)
        print(f"✓ Probabilities sum to 1")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_components():
    """Check loss and metrics."""
    print("\n" + "="*70)
    print("[6/6] Checking Training Components")
    print("="*70)
    
    try:
        from training.losses import EmotionLoss
        from training.metrics import EmotionMetrics
        import numpy as np
        
        # Test loss
        criterion = EmotionLoss(num_classes=8, label_smoothing=0.1)
        logits = torch.randn(4, 8, requires_grad=True)
        labels = torch.randint(0, 8, (4,))
        loss = criterion(logits, labels)
        loss.backward()
        print(f"✓ Loss computation: {loss.item():.4f}")
        
        # Test metrics
        metrics_calc = EmotionMetrics(num_classes=8)
        predictions = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        labels_np = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        metrics = metrics_calc.compute(predictions, labels_np)
        print(f"✓ Metrics computation:")
        print(f"  UAR: {metrics['uar']:.4f}")
        print(f"  WAR: {metrics['war']:.4f}")
        print(f"  WA-F1: {metrics['wa_f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training components check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("PIPELINE VERIFICATION")
    print("="*70)
    
    results = []
    
    # Check environment
    results.append(("Environment", check_environment()))
    
    # Check imports
    results.append(("Imports", check_imports()))
    
    # Check dataset
    results.append(("Dataset", check_dataset()))
    
    # Check model
    model_ok, model = check_model()
    results.append(("Model Creation", model_ok))
    
    # Check forward pass
    if model_ok and model is not None:
        results.append(("Forward Pass", check_forward_pass(model)))
    else:
        results.append(("Forward Pass", False))
    
    # Check training components
    results.append(("Training Components", check_training_components()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All checks passed! Pipeline is ready!")
        print("\nYou can now run:")
        print("  python scripts/train_test_samples.py")
    else:
        print("\n⚠️ Some checks failed. Please fix issues before training.")
    
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
