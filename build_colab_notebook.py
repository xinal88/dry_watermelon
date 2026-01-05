"""Build complete Colab notebook with all necessary cells."""
import json

# Base notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_markdown(text):
    """Add markdown cell."""
    lines = text.split('\n')
    # Add newline to all lines except the last one
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # Add last line without newline if not empty
        source.append(lines[-1])
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code(code):
    """Add code cell."""
    lines = code.split('\n')
    # Add newline to all lines except the last one
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # Add last line without newline if not empty
        source.append(lines[-1])
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# Title
add_markdown("""# 🎭 Multimodal FER Training - v1 (Fixed for Colab IDE)

**Improvements:**
- ✅ Auto-detect Colab environment
- ✅ Smart Google Drive mounting  
- ✅ Data validation before training
- ✅ Better error handling
- ✅ Works in both Colab web and Colab IDE extension

**Hardware**: T4 GPU (Free) or A100 (Colab Pro)
**Dataset**: RAVDESS (2968 videos)
**Expected Time**: 2-3 hours on T4""")

# Step 1: Environment Check
add_markdown("## 📋 Step 1: Environment & GPU Check")

add_code("""# Check environment
import os
import sys

IN_COLAB = 'google.colab' in sys.modules
print(f"Running in Colab: {IN_COLAB}")

if IN_COLAB:
    print("✓ Colab environment detected")
    !nvidia-smi
    
import torch
print(f"\\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")""")

# Step 2: Clone Repository
add_markdown("## 📦 Step 2: Clone Repository")

add_code("""# Clone repository if not already cloned
if IN_COLAB:
    if not os.path.exists('/content/dry_watermelon'):
        print("Cloning repository...")
        !git clone https://github.com/xinal88/dry_watermelon
        %cd dry_watermelon
    else:
        print("Repository already cloned")
        %cd dry_watermelon
        !git pull  # Update to latest
else:
    print("Not in Colab - assuming local development")""")

# Step 3: Mount Google Drive
add_markdown("""## 💾 Step 3: Mount Google Drive & Setup Data

**IMPORTANT**: Update `RAVDESS_PATH` to match your Google Drive structure""")

add_code("""# Mount Google Drive
if IN_COLAB:
    from google.colab import drive
    
    # Check if already mounted
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    else:
        print("✓ Google Drive already mounted")
    
    # Set your RAVDESS path - UPDATE THIS!
    RAVDESS_PATH = "/content/drive/MyDrive/[HUST]_Facial_Expression_Recognition/Dataset/Multimodal_DFER/RAVDESS"
    
    # Verify path exists
    if os.path.exists(RAVDESS_PATH):
        print(f"✓ RAVDESS path found: {RAVDESS_PATH}")
        
        # List ALL contents to debug
        print(f"\\nListing contents of RAVDESS folder:")
        all_items = os.listdir(RAVDESS_PATH)
        print(f"  Total items: {len(all_items)}")
        print(f"  First 10 items: {all_items[:10]}")
        
        # Check for different folder patterns
        folders = [f for f in all_items if os.path.isdir(os.path.join(RAVDESS_PATH, f))]
        actor_folders = [f for f in folders if f.startswith('Actor_')]
        video_speech_folders = [f for f in folders if f.startswith('Video_Speech_Actor_')]
        video_song_folders = [f for f in folders if f.startswith('Video_Song_Actor_')]
        
        print(f"\\nFolder analysis:")
        print(f"  Total folders: {len(folders)}")
        print(f"  Actor_* folders: {len(actor_folders)}")
        print(f"  Video_Speech_Actor_* folders: {len(video_speech_folders)}")
        print(f"  Video_Song_Actor_* folders: {len(video_song_folders)}")
        
        if actor_folders:
            print(f"  Sample Actor folders: {actor_folders[:3]}")
        if video_speech_folders:
            print(f"  Sample Video_Speech folders: {video_speech_folders[:3]}")
        
        # Check if we found any video folders
        total_video_folders = len(actor_folders) + len(video_speech_folders)
        if total_video_folders == 0:
            print(f"\\n⚠️  WARNING: No Actor_* or Video_Speech_Actor_* folders found!")
            print(f"   This might cause 'Loaded 0 videos' error later.")
            print(f"   Please check your folder structure.")
    else:
        print(f"❌ ERROR: RAVDESS path not found!")
        print(f"   Path: {RAVDESS_PATH}")
        print(f"\\n   Please update RAVDESS_PATH in this cell to match your Google Drive structure")
        raise FileNotFoundError(f"RAVDESS data not found at {RAVDESS_PATH}")
    
    # Create symlink
    if not os.path.exists('data/ravdess'):
        !ln -s {RAVDESS_PATH} data/ravdess
        print("\\n✓ Created symlink: data/ravdess -> Google Drive")
    else:
        print("\\n✓ Symlink already exists")
else:
    # Local development
    RAVDESS_PATH = "data/ravdess"
    print(f"Using local path: {RAVDESS_PATH}")""")

# Step 3.5: Reorganize Dataset
add_markdown("""## 🔄 Step 3.5: Reorganize Dataset (IMPORTANT!)

This cell will reorganize RAVDESS data into a clean structure by:
1. Finding ALL .mp4 files recursively
2. Parsing filenames to identify speech/song and actor
3. Creating organized symlinks (no copying, saves space)

**Run this cell once to fix the dataset structure!**""")

add_code("""# Reorganize RAVDESS dataset
import os
from pathlib import Path
import shutil

if IN_COLAB:
    print("="*70)
    print("REORGANIZING RAVDESS DATASET")
    print("="*70)
    
    # Create organized structure
    organized_path = Path("/content/ravdess_organized")
    organized_path.mkdir(exist_ok=True)
    
    # Find ALL .mp4 files recursively
    print(f"\\nSearching for .mp4 files in: {RAVDESS_PATH}")
    data_path = Path(RAVDESS_PATH)
    all_videos = list(data_path.rglob("*.mp4"))
    
    print(f"Found {len(all_videos)} total .mp4 files")
    
    if len(all_videos) == 0:
        print("\\n❌ ERROR: No .mp4 files found!")
        print("   Please check your RAVDESS_PATH")
        raise FileNotFoundError("No video files found")
    
    # Parse and organize by filename
    # Format: XX-YY-ZZ-AA-BB-CC-DD.mp4
    # YY: 01=speech, 02=song
    # DD: Actor ID
    
    speech_count = 0
    song_count = 0
    
    for video_file in all_videos:
        try:
            # Parse filename
            filename = video_file.name
            parts = filename.replace('.mp4', '').split('-')
            
            if len(parts) != 7:
                print(f"  Skipping invalid filename: {filename}")
                continue
            
            vocal_channel = parts[1]  # 01=speech, 02=song
            actor_id = parts[6]       # Actor number
            
            # Determine modality
            if vocal_channel == '01':
                modality = 'speech'
                speech_count += 1
            elif vocal_channel == '02':
                modality = 'song'
                song_count += 1
            else:
                print(f"  Skipping unknown modality: {filename}")
                continue
            
            # Create organized folder structure
            actor_folder = organized_path / f"Actor_{actor_id}"
            actor_folder.mkdir(exist_ok=True)
            
            # Create symlink (not copy - saves space!)
            target_link = actor_folder / filename
            
            if not target_link.exists():
                try:
                    target_link.symlink_to(video_file)
                except:
                    # If symlink fails, try copy (slower but works)
                    shutil.copy2(video_file, target_link)
        
        except Exception as e:
            print(f"  Error processing {video_file.name}: {e}")
            continue
    
    print(f"\\n✅ Reorganization complete!")
    print(f"  Speech videos: {speech_count}")
    print(f"  Song videos: {song_count}")
    print(f"  Total: {speech_count + song_count}")
    
    # List organized folders
    actor_folders = sorted([f for f in organized_path.iterdir() if f.is_dir()])
    print(f"\\n  Created {len(actor_folders)} Actor folders")
    print(f"  Sample: {[f.name for f in actor_folders[:5]]}")
    
    # Update RAVDESS_PATH to use organized structure
    RAVDESS_PATH = str(organized_path)
    print(f"\\n✅ Updated RAVDESS_PATH to: {RAVDESS_PATH}")
    print(f"✅ Dataset is now ready for training!")
    
else:
    print("Not in Colab - skipping reorganization")
    RAVDESS_PATH = "data/ravdess"

print(f"\\nFinal data path: {RAVDESS_PATH}")""")

# Step 4: Install Dependencies
add_markdown("## 🔧 Step 4: Install Dependencies")

add_code("""# Install required packages
if IN_COLAB:
    print("Installing dependencies...")
    !pip install -q transformers==4.36.0
    !pip install -q einops
    !pip install -q scikit-learn
    !pip install -q matplotlib seaborn
    !pip install -q torchaudio
    
    # Verify ffmpeg
    !which ffmpeg
    print("\\n✓ All dependencies installed!")
else:
    print("Skipping installation (local environment)")""")

# Step 5: Import Libraries
add_markdown("## 📚 Step 5: Import Libraries")

add_code("""import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.simple_ravdess_dataset import create_simple_ravdess_dataloaders

print("✓ All imports successful!")""")

# Step 6: Configuration
add_markdown("""## ⚙️ Step 6: Configuration

**Edit this section** to customize training""")

add_code("""CONFIG = {
    # ============ DATA ============
    "data_dir": RAVDESS_PATH if IN_COLAB else "data/ravdess",
    "modality": "speech",  # "speech" or "song"
    "use_audio": True,

    # ============ TRAINING ============
    "batch_size": 16,      # T4: 8-16, A100: 32-64
    "num_epochs": 40,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 2,

    # ============ MODEL (Lightweight for T4) ============
    "audio_dim": 512,
    "visual_dim": 512,
    "fusion_hidden_dim": 1024,
    "fusion_output_dim": 512,
    "num_audio_layers": 8,
    "num_visual_layers": 4,
    "num_fusion_layers": 4,

    # ============ PRETRAINED MODELS ============
    "use_pretrained_visual": False,
    "use_pretrained_fusion": False,

    # ============ OPTIMIZATION ============
    "use_amp": True,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,

    # ============ SCHEDULER ============
    "scheduler_type": "cosine",
    "warmup_epochs": 5,

    # ============ CHECKPOINTING ============
    "save_dir": "checkpoints/ravdess_speech_t4",
    "save_every": 10,
    "save_best_only": True,

    # ============ DEVICE ============
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Print configuration
print("="*70)
print("TRAINING CONFIGURATION")
print("="*70)
for key, value in CONFIG.items():
    print(f"  {key:<30}: {value}")
print("="*70)""")

# Step 7: Validate Data
add_markdown("## 🔍 Step 7: Validate Data (IMPORTANT!)")

add_code("""# Validate data before creating model
print("Validating RAVDESS dataset...")
print(f"Data directory: {CONFIG['data_dir']}")
print(f"Exists: {os.path.exists(CONFIG['data_dir'])}")

if not os.path.exists(CONFIG['data_dir']):
    raise FileNotFoundError(f"Data directory not found: {CONFIG['data_dir']}")

# Check for Actor folders
from pathlib import Path
data_path = Path(CONFIG['data_dir'])

# Look for both patterns
actor_folders = list(data_path.glob("Actor_*"))
video_speech_folders = list(data_path.glob("Video_Speech_Actor_*"))
all_folders = actor_folders + video_speech_folders

# Filter out audio-only folders
video_folders = [f for f in all_folders if f.is_dir() and not f.name.startswith("Audio_")]

print(f"\\nFound {len(video_folders)} video folders:")
if video_folders:
    print(f"  Sample folders: {[f.name for f in video_folders[:5]]}")
    
    # Check first folder structure
    sample_folder = video_folders[0]
    print(f"\\nChecking structure of: {sample_folder.name}")
    
    # Try direct .mp4 files
    videos = list(sample_folder.glob("*.mp4"))
    print(f"  Direct .mp4 files: {len(videos)}")
    
    # Try nested Actor_XX folder
    nested_folders = list(sample_folder.glob("Actor_*"))
    if nested_folders:
        print(f"  Found nested Actor folders: {len(nested_folders)}")
        nested_videos = list(nested_folders[0].glob("*.mp4"))
        print(f"  Videos in nested folder: {len(nested_videos)}")
        videos = nested_videos  # Use nested videos
    
    # Try recursive search
    if len(videos) == 0:
        print(f"  Trying recursive search...")
        videos = list(sample_folder.rglob("*.mp4"))
        print(f"  Total .mp4 files (recursive): {len(videos)}")
    
    if len(videos) > 0:
        print(f"\\n✅ Data validation PASSED!")
        print(f"✅ Found {len(videos)} videos in sample folder")
        print(f"✅ Ready to create dataloaders")
        print(f"\\nNote: Dataset loader will handle the folder structure automatically")
    else:
        print(f"\\n❌ ERROR: No .mp4 files found!")
        print(f"\\nDebugging info:")
        print(f"  Folder: {sample_folder}")
        print(f"  Contents: {list(sample_folder.iterdir())[:10]}")
        raise ValueError(f"No .mp4 files found in {sample_folder}")
else:
    raise ValueError(f"No Actor_* or Video_Speech_Actor_* folders found in {CONFIG['data_dir']}")""")

# Step 8: Create Model
add_markdown("## 🏗️ Step 8: Create Model")

add_code("""def create_model(config):
    audio_config = AudioBranchConfig(
        feature_dim=config["audio_dim"],
        num_layers=config["num_audio_layers"],
        num_segments=8,
    )

    visual_config = VisualBranchConfig(
        use_pretrained_encoder=config["use_pretrained_visual"],
        feature_dim=config["visual_dim"],
        temporal_depth=config["num_visual_layers"],
    )

    fusion_config = LFM2FusionConfig(
        use_pretrained=config["use_pretrained_fusion"],
        num_layers=config["num_fusion_layers"],
        hidden_dim=config["fusion_hidden_dim"],
        audio_dim=config["audio_dim"],
        visual_dim=config["visual_dim"],
        output_dim=config["fusion_output_dim"],
    )

    model = MultimodalFER(
        audio_config=audio_config,
        visual_config=visual_config,
        fusion_config=fusion_config,
        num_classes=8,
        num_segments=8,
    )

    return model

print("Creating model...")
model = create_model(CONFIG)
model = model.to(CONFIG["device"])

model.print_summary()""")

# Step 9: Create Dataloaders
add_markdown("## 📊 Step 9: Create Dataloaders")

add_code("""print("Creating dataloaders...")
print(f"Data directory: {CONFIG['data_dir']}")

try:
    train_loader, val_loader, test_loader = create_simple_ravdess_dataloaders(
        data_dir=CONFIG["data_dir"],
        modality=CONFIG["modality"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        use_audio=CONFIG["use_audio"],
    )
    
    print(f"\\n✅ Dataloaders created successfully!")
    print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} samples ({len(test_loader)} batches)")
    
    if len(train_loader.dataset) == 0:
        raise ValueError("Train dataset is empty! Check data directory and folder structure.")
        
except Exception as e:
    print(f"\\n❌ ERROR creating dataloaders:")
    print(f"   {str(e)}")
    print(f"\\nDebugging info:")
    print(f"   Data dir: {CONFIG['data_dir']}")
    print(f"   Exists: {os.path.exists(CONFIG['data_dir'])}")
    if os.path.exists(CONFIG['data_dir']):
        folders = os.listdir(CONFIG['data_dir'])
        print(f"   Contents: {folders[:10]}")
    raise""")

# Step 10: Training Setup
add_markdown("## 🎯 Step 10: Training Setup")

add_code("""# Loss and metrics
criterion = EmotionLoss()
metrics_calculator = EmotionMetrics(num_classes=8)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

# Scheduler
if CONFIG["scheduler_type"] == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"] - CONFIG["warmup_epochs"],
        eta_min=CONFIG["lr"] * 0.01,
    )
else:
    scheduler = None

# Mixed precision scaler
scaler = GradScaler() if CONFIG["use_amp"] else None

# Training history
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": [],
}

print("✓ Training setup complete!")
print(f"  Optimizer: AdamW (lr={CONFIG['lr']})")
print(f"  Scheduler: {CONFIG['scheduler_type']}")
print(f"  Mixed Precision: {CONFIG['use_amp']}")""")

# Step 11: Training Functions
add_markdown("## 🔄 Step 11: Training Functions")

add_code("""def train_epoch(model, train_loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (audio, video, labels, _) in enumerate(pbar):
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])
        
        optimizer.zero_grad()
        
        if config["use_amp"]:
            with autocast():
                outputs = model(audio, video)
                # Extract logits if output is dict
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            if config["max_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(audio, video)
            # Extract logits if output is dict
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, metrics_calculator, config):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for audio, video, labels, _ in tqdm(val_loader, desc="Validation"):
            audio = audio.to(config["device"])
            video = video.to(config["device"])
            labels = labels.to(config["device"])
            
            if config["use_amp"]:
                with autocast():
                    outputs = model(audio, video)
                    # Extract logits if output is dict
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss = criterion(logits, labels)
            else:
                outputs = model(audio, video)
                # Extract logits if output is dict
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = metrics_calculator.compute(
        np.array(all_predictions),
        np.array(all_labels)
    )
    
    return total_loss / len(val_loader), metrics

print("✓ Training functions defined")""")

# Step 12: Main Training Loop
add_markdown("""## 🚀 Step 12: Main Training Loop

**This will take 2-3 hours on T4 GPU**""")

add_code("""# Create save directory
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# Save config
with open(f"{CONFIG['save_dir']}/config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

best_val_acc = 0.0
start_time = datetime.now()

print("="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total epochs: {CONFIG['num_epochs']}")
print(f"Save directory: {CONFIG['save_dir']}")
print("="*70)

for epoch in range(1, CONFIG["num_epochs"] + 1):
    print(f"\\nEpoch {epoch}/{CONFIG['num_epochs']}")
    print("-" * 70)
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler, CONFIG
    )
    
    # Validate
    val_loss, val_metrics = validate(
        model, val_loader, criterion, metrics_calculator, CONFIG
    )
    
    # Update scheduler
    if scheduler is not None and epoch > CONFIG["warmup_epochs"]:
        scheduler.step()
    
    # Record history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_metrics["accuracy"])
    history["val_f1"].append(val_metrics["macro_f1"])
    
    # Print metrics
    print(f"\\nResults:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
    print(f"  Val F1: {val_metrics['macro_f1']:.4f}")
    
    # Save checkpoint
    is_best = val_metrics["accuracy"] > best_val_acc
    if is_best:
        best_val_acc = val_metrics["accuracy"]
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
            'config': CONFIG,
        }, f"{CONFIG['save_dir']}/best_model.pt")
        print(f"  ✓ New best model! Saved to: {CONFIG['save_dir']}/best_model.pt")
    
    if epoch % CONFIG["save_every"] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_metrics["accuracy"],
            'config': CONFIG,
        }, f"{CONFIG['save_dir']}/checkpoint_epoch_{epoch}.pt")
        print(f"  ✓ Checkpoint saved: checkpoint_epoch_{epoch}.pt")

# Save final model
torch.save({
    'epoch': CONFIG["num_epochs"],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_metrics["accuracy"],
    'config': CONFIG,
}, f"{CONFIG['save_dir']}/final_model.pt")

# Save history
with open(f"{CONFIG['save_dir']}/training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

end_time = datetime.now()
duration = end_time - start_time

print("\\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Duration: {duration}")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Checkpoints saved to: {CONFIG['save_dir']}")
print("="*70)""")

# Step 13: Plot Results
add_markdown("## 📈 Step 13: Plot Training Curves")

add_code("""fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(history["train_loss"], label="Train")
axes[0, 0].plot(history["val_loss"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss Curves")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy curves
axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].set_title("Accuracy Curves")
axes[0, 1].legend()
axes[0, 1].grid(True)

# F1 score
axes[1, 0].plot(history["val_f1"])
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("F1 Score")
axes[1, 0].set_title("Validation F1 Score")
axes[1, 0].grid(True)

# Summary
axes[1, 1].axis('off')
summary_text = f'''
Training Summary
{'='*30}

Best Val Accuracy: {best_val_acc:.2f}%
Final Train Acc: {history["train_acc"][-1]:.2f}%
Final Val Acc: {history["val_acc"][-1]:.2f}%
Final Val F1: {history["val_f1"][-1]:.4f}

Total Epochs: {CONFIG["num_epochs"]}
Batch Size: {CONFIG["batch_size"]}
Learning Rate: {CONFIG["lr"]}

Duration: {duration}
'''
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace')

plt.tight_layout()
plt.savefig(f"{CONFIG['save_dir']}/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Plot saved to: {CONFIG['save_dir']}/training_curves.png")""")

# Step 14: Test Evaluation
add_markdown("## 🧪 Step 14: Evaluate on Test Set")

add_code("""print("="*70)
print("EVALUATING ON TEST SET")
print("="*70)

# Load best model
checkpoint = torch.load(f"{CONFIG['save_dir']}/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")

# Evaluate
test_loss, test_metrics = validate(
    model, test_loader, criterion, metrics_calculator, CONFIG
)

print(f"\\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
print(f"  F1 Score: {test_metrics['macro_f1']:.4f}")
print(f"  UAR (Macro Recall): {test_metrics['uar']:.4f}")
print(f"  WAR (Weighted Recall): {test_metrics['war']:.4f}")

# Save test results
test_results = {
    "test_loss": test_loss,
    "test_metrics": test_metrics,
}

with open(f"{CONFIG['save_dir']}/test_results.json", 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\\n✓ Test results saved to: {CONFIG['save_dir']}/test_results.json")""")

# Step 15: Download
add_markdown("""## 💾 Step 15: Download Checkpoints

Run this cell to download trained models to your local machine""")

add_code("""if IN_COLAB:
    from google.colab import files
    
    print("Preparing files for download...")
    
    # Zip checkpoints
    !zip -r checkpoints.zip {CONFIG["save_dir"]}
    
    print("\\nDownloading checkpoints.zip...")
    files.download('checkpoints.zip')
    
    print("✓ Download complete!")
else:
    print("Not in Colab - checkpoints are already local")
    print(f"Location: {CONFIG['save_dir']}")""")

# Final message
add_markdown("""## 🎉 Training Complete!

**Next Steps:**
1. ✅ Download checkpoints (above)
2. ✅ Use `best_model.pt` for inference
3. ✅ Check `training_curves.png` for visualization
4. ✅ Review `test_results.json` for final metrics

**Files Created:**
- `best_model.pt` - Best validation accuracy
- `final_model.pt` - Last epoch
- `checkpoint_epoch_*.pt` - Periodic checkpoints
- `training_history.json` - All metrics
- `training_curves.png` - Visualization
- `test_results.json` - Test set performance
- `config.json` - Training configuration

**Congratulations! 🎊**""")

# Save notebook
with open('train_dry_watermelon_v1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Created notebook with {len(notebook['cells'])} cells")
print("✅ Saved to: train_dry_watermelon_v1.ipynb")
