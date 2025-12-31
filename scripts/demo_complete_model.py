"""
Demo Complete Multimodal FER Model
Demonstrates the full pipeline with dummy data
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main demo function."""
    print("="*70)
    print("Complete Multimodal FER Model Demo")
    print("="*70)
    
    from models import MultimodalFER
    
    # Create model
    print("\n[1/6] Creating model...")
    model = MultimodalFER(
        num_classes=8,
        num_segments=8,
    )
    
    # Print summary
    print("\n[2/6] Model summary:")
    model.print_summary()
    
    # Create dummy data
    print("\n[3/6] Creating dummy data...")
    batch_size = 2
    
    # Audio: 3 seconds at 16kHz
    audio = torch.randn(batch_size, 48000)
    print(f"  Audio: {audio.shape}")
    
    # Video: 16 frames, 224x224
    video = torch.randn(batch_size, 16, 3, 224, 224)
    print(f"  Video: {video.shape}")
    
    # Labels
    labels = torch.randint(0, 8, (batch_size,))
    print(f"  Labels: {labels}")
    
    # Forward pass
    print("\n[4/6] Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            audio=audio,
            video=video,
            return_intermediates=True,
        )
    
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Probabilities: {outputs['probabilities'].shape}")
    
    # Show predictions
    print("\n[5/6] Predictions:")
    emotion_labels = model.get_emotion_labels()
    
    for i in range(batch_size):
        probs = outputs['probabilities'][i]
        pred_idx = probs.argmax().item()
        pred_emotion = emotion_labels[pred_idx]
        confidence = probs[pred_idx].item()
        
        print(f"\n  Sample {i+1}:")
        print(f"    True label: {emotion_labels[labels[i]]}")
        print(f"    Predicted: {pred_emotion} (confidence: {confidence:.2%})")
        
        # Top-3 predictions
        top3_probs, top3_indices = torch.topk(probs, 3)
        print(f"    Top-3:")
        for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"      {j+1}. {emotion_labels[idx]}: {prob:.2%}")
    
    # Test training step
    print("\n[6/6] Testing training step...")
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Forward
    outputs = model(audio, video)
    loss = criterion(outputs["logits"], labels)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"  Gradient norm: {total_grad_norm:.4f}")
    
    # Optimizer step
    optimizer.step()
    print(f"  ✓ Training step successful")
    
    # Test modality-specific forward passes
    print("\n" + "="*70)
    print("Testing Modality-Specific Forward Passes")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Audio only
        print("\n[Audio Only]")
        audio_outputs = model.forward_audio_only(audio)
        audio_pred = audio_outputs['probabilities'].argmax(dim=1)
        print(f"  Predictions: {[emotion_labels[i] for i in audio_pred]}")
        
        # Visual only
        print("\n[Visual Only]")
        visual_outputs = model.forward_visual_only(video)
        visual_pred = visual_outputs['probabilities'].argmax(dim=1)
        print(f"  Predictions: {[emotion_labels[i] for i in visual_pred]}")
        
        # Multimodal
        print("\n[Multimodal]")
        multi_outputs = model(audio, video)
        multi_pred = multi_outputs['probabilities'].argmax(dim=1)
        print(f"  Predictions: {[emotion_labels[i] for i in multi_pred]}")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)
    
    # Show next steps
    print("\n📚 Next Steps:")
    print("  1. Implement RAVDESS dataset loader")
    print("  2. Create training pipeline")
    print("  3. Train on real data")
    print("  4. Evaluate performance")
    print("\n  See TRAINING_GUIDE.md for detailed instructions")
    print("="*70)


if __name__ == "__main__":
    main()
