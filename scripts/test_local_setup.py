"""
Test local setup before training

Run this to verify everything is working:
    python scripts/test_local_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import warnings
warnings.filterwarnings('ignore')


def test_environment():
    """Test Python and PyTorch environment."""
    print("="*70)
    print("1. ENVIRONMENT CHECK")
    print("="*70)
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = "cuda"
    else:
        print("⚠️  No GPU found, will use CPU (slower)")
        device = "cpu"
    
    print(f"✓ Environment OK\n")
    return device


def test_data():
    """Test data directory."""
    print("="*70)
    print("2. DATA CHECK")
    print("="*70)
    
    data_path = Path("data/ravdess")
    print(f"Data directory: {data_path}")
    print(f"Exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("❌ Data directory not found!")
        print("   Please ensure RAVDESS data is in data/ravdess/")
        return False
    
    speech_folders = list(data_path.glob("Video_Speech_Actor_*"))
    song_folders = list(data_path.glob("Video_Song_Actor_*"))
    
    print(f"Speech actors: {len(speech_folders)}")
    print(f"Song actors: {len(song_folders)}")
    
    if len(speech_folders) == 0:
        print("❌ No Video_Speech_Actor folders found!")
        return False
    
    # Check one folder for videos
    sample_folder = speech_folders[0]
    actor_num = sample_folder.name.split("_")[-1]
    nested_folder = sample_folder / f"Actor_{actor_num}"
    
    if nested_folder.exists():
        videos = list(nested_folder.glob("*.mp4"))
        print(f"Sample folder: {sample_folder.name}/Actor_{actor_num}")
    else:
        videos = list(sample_folder.glob("*.mp4"))
        print(f"Sample folder: {sample_folder.name}")
    
    print(f"Videos in folder: {len(videos)}")
    
    if len(videos) == 0:
        print("❌ No .mp4 files found in actor folders!")
        return False
    
    print(f"✓ Data OK\n")
    return True


def test_imports():
    """Test project imports."""
    print("="*70)
    print("3. IMPORTS CHECK")
    print("="*70)
    
    try:
        from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
        print("✓ Model imports OK")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from training.losses import EmotionLoss
        from training.metrics import EmotionMetrics
        print("✓ Training imports OK")
    except Exception as e:
        print(f"❌ Training import failed: {e}")
        return False
    
    try:
        from data.ravdess_dataset import create_ravdess_dataloaders
        print("✓ Data imports OK")
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    print(f"✓ All imports OK\n")
    return True


def test_dataloader():
    """Test dataloader creation."""
    print("="*70)
    print("4. DATALOADER CHECK")
    print("="*70)
    
    try:
        from data.ravdess_dataset import create_ravdess_dataloaders
        
        print("Creating dataloaders (this may take a moment)...")
        train_loader, val_loader, test_loader = create_ravdess_dataloaders(
            data_dir="data/ravdess",
            modality="speech",
            batch_size=2,
            num_workers=0,  # 0 for testing
            use_audio=True,
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        if len(train_loader.dataset) == 0:
            print("❌ No training samples loaded!")
            return False
        
        # Test loading one batch
        print("\nTesting batch loading...")
        audio, video, labels, metadata = next(iter(train_loader))
        
        print(f"Audio shape: {audio.shape}")
        print(f"Video shape: {video.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample emotion: {metadata['emotion'][0]}")
        
        print(f"✓ Dataloader OK\n")
        return True
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(device):
    """Test model creation and forward pass."""
    print("="*70)
    print("5. MODEL CHECK")
    print("="*70)
    
    try:
        from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
        
        print("Creating model...")
        audio_config = AudioBranchConfig(feature_dim=512, num_layers=4)
        visual_config = VisualBranchConfig(
            use_pretrained_encoder=False,  # Use custom encoder
            feature_dim=512,
            temporal_depth=2
        )
        fusion_config = LFM2FusionConfig(
            use_pretrained=False,  # Use custom LFM2
            num_layers=2,
            hidden_dim=512,
            audio_dim=512,
            visual_dim=512,
            output_dim=512,
        )
        
        model = MultimodalFER(
            audio_config=audio_config,
            visual_config=visual_config,
            fusion_config=fusion_config,
            num_classes=8,
        )
        
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 2
        audio = torch.randn(batch_size, 48000).to(device)  # 3 seconds @ 16kHz
        video = torch.randn(batch_size, 16, 3, 224, 224).to(device)  # 16 frames
        
        with torch.no_grad():
            outputs = model(audio, video)
        
        # Handle dict or tensor output
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('output', None))
            if logits is None:
                print(f"Output keys: {outputs.keys()}")
                logits = list(outputs.values())[0]
        else:
            logits = outputs
        
        print(f"Input audio: {audio.shape}")
        print(f"Input video: {video.shape}")
        print(f"Output: {logits.shape}")
        
        assert logits.shape == (batch_size, 8), f"Expected shape (2, 8), got {logits.shape}"
        
        print(f"✓ Model OK\n")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LOCAL SETUP TEST")
    print("="*70 + "\n")
    
    # Run tests
    device = test_environment()
    
    data_ok = test_data()
    if not data_ok:
        print("\n❌ SETUP FAILED: Data issues")
        print("   Fix data directory and try again")
        return
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ SETUP FAILED: Import issues")
        print("   Check dependencies installation")
        return
    
    dataloader_ok = test_dataloader()
    if not dataloader_ok:
        print("\n❌ SETUP FAILED: Dataloader issues")
        return
    
    model_ok = test_model(device)
    if not model_ok:
        print("\n❌ SETUP FAILED: Model issues")
        return
    
    # All tests passed
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYou're ready to train! Run:")
    print("  python scripts/train_ravdess.py --data_dir data/ravdess --epochs 100")
    print("\nOr for a quick test:")
    print("  python scripts/train_ravdess.py --data_dir data/ravdess --epochs 1 --batch_size 2")


if __name__ == "__main__":
    main()
