"""
Quick Pipeline Test

Test all components before training:
1. Dataset loader
2. Model forward pass
3. Loss computation
4. Metrics calculation
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_dataset():
    """Test dataset loader."""
    print("\n" + "="*70)
    print("[1/4] Testing Dataset Loader")
    print("="*70)
    
    from data.test_dataset import RAVDESSTestDataset, create_test_dataloader
    
    try:
        # Create dataset
        dataset = RAVDESSTestDataset(data_dir="data/test_samples")
        print(f"[OK] Dataset created: {len(dataset)} samples")
        
        # Load first sample
        audio, video, label, metadata = dataset[0]
        print(f"[OK] Sample loaded:")
        print(f"  Audio: {audio.shape}")
        print(f"  Video: {video.shape}")
        print(f"  Label: {label} ({dataset.emotion_names[label]})")
        
        # Create dataloader
        dataloader = create_test_dataloader(batch_size=2)
        print(f"[OK] Dataloader created: {len(dataloader)} batches")
        
        # Test batch
        for audio_batch, video_batch, labels_batch, _ in dataloader:
            print(f"[OK] Batch loaded:")
            print(f"  Audio: {audio_batch.shape}")
            print(f"  Video: {video_batch.shape}")
            print(f"  Labels: {labels_batch}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "="*70)
    print("[2/4] Testing Model Forward Pass")
    print("="*70)
    
    from models import MultimodalFER
    
    try:
        # Create model
        model = MultimodalFER(num_classes=8, num_segments=8)
        print(f"[OK] Model created")
        
        # Create dummy inputs
        batch_size = 2
        audio = torch.randn(batch_size, 48000)
        video = torch.randn(batch_size, 16, 3, 224, 224)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(audio, video)
        
        print(f"[OK] Forward pass successful:")
        print(f"  Logits: {outputs['logits'].shape}")
        print(f"  Probabilities: {outputs['probabilities'].shape}")
        
        # Check probabilities sum to 1
        prob_sum = outputs['probabilities'].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-5)
        print(f"[OK] Probabilities sum to 1")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test loss computation."""
    print("\n" + "="*70)
    print("[3/4] Testing Loss Computation")
    print("="*70)
    
    from training.losses import EmotionLoss
    
    try:
        # Create loss
        criterion = EmotionLoss(num_classes=8, label_smoothing=0.1)
        print(f"[OK] Loss function created")
        
        # Create dummy data
        batch_size = 4
        logits = torch.randn(batch_size, 8, requires_grad=True)  # Enable gradient
        labels = torch.randint(0, 8, (batch_size,))
        
        # Compute loss
        loss = criterion(logits, labels)
        print(f"[OK] Loss computed: {loss.item():.4f}")
        
        # Test backward
        loss.backward()
        print(f"[OK] Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\n" + "="*70)
    print("[4/4] Testing Metrics Calculation")
    print("="*70)
    
    from training.metrics import EmotionMetrics
    import numpy as np
    
    try:
        # Create metrics calculator
        metrics_calc = EmotionMetrics(num_classes=8)
        print(f"[OK] Metrics calculator created")
        
        # Create dummy predictions
        predictions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 0])
        
        # Compute metrics
        metrics = metrics_calc.compute(predictions, labels)
        print(f"[OK] Metrics computed:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  UAR: {metrics['uar']:.4f}")
        print(f"  WAR: {metrics['war']:.4f}")
        print(f"  WA-F1: {metrics['wa_f1']:.4f}")
        
        # Print detailed metrics
        metrics_calc.print_metrics(metrics)
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test complete training step."""
    print("\n" + "="*70)
    print("[BONUS] Testing Complete Training Step")
    print("="*70)
    
    from models import MultimodalFER
    from training.losses import EmotionLoss
    
    try:
        # Create model and loss
        model = MultimodalFER(num_classes=8, num_segments=8)
        criterion = EmotionLoss(num_classes=8, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print(f"[OK] Training setup created")
        
        # Create dummy batch
        batch_size = 2
        audio = torch.randn(batch_size, 48000)
        video = torch.randn(batch_size, 16, 3, 224, 224)
        labels = torch.randint(0, 8, (batch_size,))
        
        # Training step
        model.train()
        
        # Forward
        outputs = model(audio, video)
        loss = criterion(outputs["logits"], labels)
        print(f"[OK] Forward pass: loss = {loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        print(f"[OK] Backward pass successful")
        
        # Optimizer step
        optimizer.step()
        print(f"[OK] Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PIPELINE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Dataset Loader", test_dataset),
        ("Model Forward Pass", test_model),
        ("Loss Computation", test_loss),
        ("Metrics Calculation", test_metrics),
        ("Training Step", test_training_step),
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Ready to train!")
        print("\nNext steps:")
        print("  1. python scripts/train_test_samples.py")
        print("  2. python scripts/evaluate.py --checkpoint checkpoints/test_samples/best_model.pth")
    else:
        print("\n[WARNING] Some tests failed. Please fix issues before training.")
    
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
