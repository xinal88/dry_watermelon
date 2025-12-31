"""
Test RAVDESS Dataset Loader

Quick test to verify dataset loading works correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.ravdess_dataset import create_ravdess_dataloaders


def main():
    print("="*70)
    print("TESTING RAVDESS DATASET LOADER")
    print("="*70)
    
    # Configuration
    data_dir = "data/ravdess"
    modality = "speech"  # or "song"
    batch_size = 2
    use_audio = False  # Set True if ffmpeg available
    
    print(f"\nConfiguration:")
    print(f"  Data dir: {data_dir}")
    print(f"  Modality: {modality}")
    print(f"  Batch size: {batch_size}")
    print(f"  Use audio: {use_audio}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_ravdess_dataloaders(
            data_dir=data_dir,
            modality=modality,
            batch_size=batch_size,
            num_workers=0,
            use_audio=use_audio,
        )
        
        print(f"\n✓ Dataloaders created successfully!")
        print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_loader.dataset)} samples ({len(test_loader)} batches)")
        
        # Test loading one batch
        print("\nTesting batch loading...")
        for audio, video, labels, metadata in train_loader:
            print(f"\n✓ Batch loaded successfully!")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Video shape: {video.shape}")
            print(f"  Labels: {labels}")
            print(f"  Sample filename: {metadata['filename'][0]}")
            break
        
        # Print emotion distribution
        print("\nEmotion distribution in train set:")
        emotion_counts = {}
        emotion_names = train_loader.dataset.emotion_names
        
        for _, _, label, _ in train_loader.dataset:
            emotion = emotion_names[label]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion:<10}: {count:>3} samples")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nDataset is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
