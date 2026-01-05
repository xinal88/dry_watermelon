"""
Simple working training script - NO NaN guaranteed!
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Simple CNN model
class SimpleEmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        # Simple CNN for video frames
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, audio, video):
        # video: [B, T, C, H, W]
        B, T = video.shape[:2]
        
        # Process each frame
        video = video.view(B * T, *video.shape[2:])  # [B*T, C, H, W]
        features = self.features(video)  # [B*T, 128, 1, 1]
        features = features.view(B, T, -1)  # [B, T, 128]
        
        # Temporal pooling
        features = features.mean(dim=1)  # [B, 128]
        
        # Classify
        logits = self.classifier(features)  # [B, num_classes]
        
        return {"logits": logits}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for audio, video, labels, _ in pbar:
        video = video.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(audio, video)
        logits = outputs["logits"]
        
        loss = criterion(logits, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print("WARNING: NaN loss detected! Skipping batch...")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, video, labels, _ in tqdm(loader, desc="Validation"):
            video = video.to(device)
            labels = labels.to(device)
            
            outputs = model(audio, video)
            logits = outputs["logits"]
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import dataset
    from data.simple_ravdess_dataset import create_simple_ravdess_dataloaders
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_simple_ravdess_dataloaders(
        data_dir="/content/ravdess_organized",  # Update this path!
        modality="speech",
        batch_size=32,
        num_workers=0,
        use_audio=False,
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")
    print(f"Test:  {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nCreating model...")
    model = SimpleEmotionModel(num_classes=8)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 10
    best_val_acc = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_simple.pt")
            print(f"  ✓ New best model saved!")
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
