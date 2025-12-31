"""
Training Script for Test Samples - LIGHTWEIGHT VERSION

Optimized for 4GB VRAM (RTX 3050 Laptop)
Uses custom implementations instead of pretrained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig
from training.losses import EmotionLoss
from training.metrics import EmotionMetrics
from data.test_dataset import create_test_dataloader


class Trainer:
    """Simple trainer for test samples."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        num_epochs: int = 50,
        save_dir: str = "checkpoints",
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Loss function (Primary: CrossEntropy + Label Smoothing)
        self.criterion = EmotionLoss(
            num_classes=8,
            label_smoothing=0.1,
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        # Mixed precision (only for CUDA)
        self.use_amp = device == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Metrics
        self.metrics_calculator = EmotionMetrics(num_classes=8)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_uar": [],
            "val_war": [],
            "val_wa_f1": [],
        }
        
        self.best_uar = 0.0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for audio, video, labels, _ in pbar:
            audio = audio.to(self.device)
            video = video.to(self.device)
            labels = labels.to(self.device)
            
            # Forward with mixed precision (CUDA only)
            if self.use_amp:
                with autocast():
                    outputs = self.model(audio, video)
                    loss = self.criterion(outputs["logits"], labels)
            else:
                outputs = self.model(audio, video)
                loss = self.criterion(outputs["logits"], labels)
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for audio, video, labels, _ in self.val_loader:
            audio = audio.to(self.device)
            video = video.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(audio, video)
            loss = self.criterion(outputs["logits"], labels)
            
            # Get predictions
            predictions = outputs["probabilities"].argmax(dim=1)
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = self.metrics_calculator.compute(
            predictions=all_predictions,
            labels=all_labels,
        )
        
        metrics["loss"] = total_loss / num_batches
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("TRAINING ON TEST SAMPLES (LIGHTWEIGHT)")
        print("="*70)
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("="*70)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_uar"].append(val_metrics["uar"])
            self.history["val_war"].append(val_metrics["war"])
            self.history["val_wa_f1"].append(val_metrics["wa_f1"])
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  UAR:        {val_metrics['uar']:.4f}")
            print(f"  WAR:        {val_metrics['war']:.4f}")
            print(f"  WA-F1:      {val_metrics['wa_f1']:.4f}")
            
            # Save best model
            if val_metrics["uar"] > self.best_uar:
                self.best_uar = val_metrics["uar"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ Best model saved (UAR: {self.best_uar:.4f})")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Save final model
        self.save_checkpoint(self.num_epochs - 1, val_metrics, is_best=False, final=True)
        
        # Save training history
        self.save_history()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best UAR: {self.best_uar:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }
        
        if is_best:
            path = self.save_dir / "best_model.pth"
        elif final:
            path = self.save_dir / "final_model.pth"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")


def main():
    """Main training function."""
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device = "cuda"
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("\n⚠️ CUDA not available, using CPU")
        print("  Training will be slower on CPU")
    
    # Configuration
    config = {
        "data_dir": "data/test_samples",
        "batch_size": 1,  # Small batch for 4GB VRAM
        "num_workers": 0,
        "lr": 1e-4,
        "num_epochs": 50,
        "device": device,
        "save_dir": "checkpoints/test_samples_lightweight",
    }
    
    print("\n" + "="*70)
    print("MULTIMODAL FER - LIGHTWEIGHT TRAINING")
    print("="*70)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model with LIGHTWEIGHT config
    print("\n[1/4] Creating lightweight model...")
    print("  Using custom implementations (no pretrained downloads)")
    
    # Lightweight configs
    visual_config = VisualBranchConfig(
        use_pretrained_encoder=False,  # Custom CNN
        feature_dim=512,  # Reduced from 768
        temporal_depth=4,  # Reduced from 6
    )
    
    fusion_config = LFM2FusionConfig(
        use_pretrained=False,  # Custom LFM2
        num_layers=4,  # Reduced from 6
        hidden_dim=1024,  # Reduced from 1536
        audio_dim=512,
        visual_dim=512,
    )
    
    model = MultimodalFER(
        visual_config=visual_config,
        fusion_config=fusion_config,
        num_classes=8,
        num_segments=8,
    )
    
    print("  ✓ Model created successfully")
    
    # Print model summary
    model.print_summary()
    
    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    
    train_loader = create_test_dataloader(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    
    val_loader = create_test_dataloader(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create trainer
    print("\n[3/4] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        lr=config["lr"],
        num_epochs=config["num_epochs"],
        save_dir=config["save_dir"],
    )
    
    # Train
    print("\n[4/4] Starting training...")
    trainer.train()
    
    print("\n✓ Training completed successfully!")
    print(f"  Best UAR: {trainer.best_uar:.4f}")
    print(f"  Checkpoints: {config['save_dir']}")


if __name__ == "__main__":
    main()
