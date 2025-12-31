"""Quick test to verify the forward pass works"""
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER

print("Creating model...")
model = MultimodalFER(num_classes=8, num_segments=8)

print("\nTesting forward pass...")
audio = torch.randn(1, 48000)
video = torch.randn(1, 16, 3, 224, 224)

try:
    output = model(audio, video)
    print("✓ Forward pass successful!")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Probabilities shape: {output['probabilities'].shape}")
    print("\n✅ Model is working correctly!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
