"""
Demo script for Audio Branch
Demonstrates the complete audio processing pipeline
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.audio_branch import AudioBranch, AudioBranchConfig
import matplotlib.pyplot as plt
import numpy as np


def create_demo_audio(duration=3.0, sample_rate=16000):
    """Create a demo audio signal (sine wave with noise)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies (simulating speech-like signal)
    signal = (
        np.sin(2 * np.pi * 200 * t) +  # 200 Hz
        0.5 * np.sin(2 * np.pi * 400 * t) +  # 400 Hz
        0.3 * np.sin(2 * np.pi * 800 * t) +  # 800 Hz
        0.1 * np.random.randn(len(t))  # Noise
    )
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return torch.tensor(signal, dtype=torch.float32)


def visualize_audio_features(audio, segment_features, encoder_features):
    """Visualize audio and extracted features"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Raw audio waveform
    axes[0].plot(audio.numpy())
    axes[0].set_title("Raw Audio Waveform", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Encoder features (temporal)
    encoder_np = encoder_features[0].detach().numpy()  # [T, D]
    im1 = axes[1].imshow(encoder_np.T, aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title(f"FastConformer Features [T={encoder_np.shape[0]}, D={encoder_np.shape[1]}]", 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Feature Dimension")
    plt.colorbar(im1, ax=axes[1])
    
    # Plot 3: Segment features
    segment_np = segment_features[0].detach().numpy()  # [S, D]
    im2 = axes[2].imshow(segment_np.T, aspect='auto', cmap='plasma', origin='lower')
    axes[2].set_title(f"Segment Features [S={segment_np.shape[0]}, D={segment_np.shape[1]}]", 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Segment")
    axes[2].set_ylabel("Feature Dimension")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    # Save figure
    output_path = project_root / "audio_branch_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    # plt.show()  # Uncomment to display


def main():
    """Main demo function"""
    print("=" * 70)
    print("Audio Branch Demo - FastConformer + Segment Attention Pooling")
    print("=" * 70)
    
    # Configuration
    config = AudioBranchConfig(
        pretrained_model=None,  # Use custom implementation
        feature_dim=512,
        use_nemo=False,
        num_layers=4,  # Lightweight for demo
        d_model=512,
        num_heads=8,
        num_segments=8,
        pooling_type="attention",
        sample_rate=16000,
        n_mels=80
    )
    
    print("\n📋 Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Create audio branch
    print("\n🏗️  Building Audio Branch...")
    audio_branch = AudioBranch(**config.to_dict())
    
    # Get model info
    params = audio_branch.get_num_params()
    print(f"\n📊 Model Statistics:")
    print(f"  Total Parameters: {params['total']:,}")
    print(f"  Trainable Parameters: {params['trainable']:,}")
    print(f"  Encoder Parameters: {params['encoder']:,}")
    print(f"  Pooling Parameters: {params['pooling']:,}")
    
    # Create demo audio
    print("\n🎵 Creating demo audio signal...")
    batch_size = 2
    duration = 3.0
    sample_rate = 16000
    
    audio_batch = torch.stack([
        create_demo_audio(duration, sample_rate) for _ in range(batch_size)
    ])
    
    print(f"  Audio shape: {audio_batch.shape}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate}Hz")
    
    # Forward pass
    print("\n⚙️  Running forward pass...")
    audio_branch.eval()
    with torch.no_grad():
        output = audio_branch(audio_batch, return_all_segments=True)
    
    # Display results
    print("\n✅ Output:")
    print(f"  Segment Features: {output['segment_features'].shape}")
    print(f"  Encoder Features: {output['encoder_features'].shape}")
    print(f"  Number of Segments: {output['num_segments']}")
    
    # Compute statistics
    segment_feats = output['segment_features']
    print(f"\n📈 Feature Statistics:")
    print(f"  Mean: {segment_feats.mean().item():.4f}")
    print(f"  Std: {segment_feats.std().item():.4f}")
    print(f"  Min: {segment_feats.min().item():.4f}")
    print(f"  Max: {segment_feats.max().item():.4f}")
    
    # Visualize
    print("\n🎨 Creating visualization...")
    visualize_audio_features(
        audio_batch[0],
        output['segment_features'],
        output['encoder_features']
    )
    
    # Test different pooling modes
    print("\n🔄 Testing single representation mode...")
    with torch.no_grad():
        output_single = audio_branch(audio_batch, return_all_segments=False)
    print(f"  Single representation shape: {output_single['segment_features'].shape}")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)
    
    return audio_branch, output


if __name__ == "__main__":
    audio_branch, output = main()

