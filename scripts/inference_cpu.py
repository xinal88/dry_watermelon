"""
Inference Script for CPU-trained Model

Easy configuration at the top for quick testing.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER, VisualBranchConfig, LFM2FusionConfig, AudioBranchConfig

# ============================================================================
# 🎯 CONFIGURATION - EDIT THIS SECTION
# ============================================================================

CONFIG = {
    # Model checkpoint
    "checkpoint_path": "checkpoints/test_samples_cpu/best_model.pth",
    
    # Video to test (change this to test different videos)
    "video_path": "data/test_samples/01-02-01-01-01-01-01.mp4",
    
    # Device
    "device": "cpu",  # Use "cuda" if you have GPU
    
    # Model configuration (must match training config)
    "audio_dim": 256,
    "visual_dim": 256,
    "fusion_hidden_dim": 512,
    "fusion_output_dim": 512,
    "num_audio_layers": 4,
    "num_visual_layers": 2,
    "num_fusion_layers": 2,
    
    # Display options
    "show_all_probs": True,  # Show all emotion probabilities
    "show_top_k": 3,  # Show top-K predictions
}

# Emotion labels (RAVDESS)
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# ============================================================================
# 🎬 INFERENCE CODE
# ============================================================================

class EmotionPredictor:
    """Predict emotion from video."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config["device"]
        self.emotion_labels = EMOTION_LABELS
        
        print("="*70)
        print("EMOTION RECOGNITION - INFERENCE")
        print("="*70)
        print(f"Checkpoint: {config['checkpoint_path']}")
        print(f"Device: {self.device}")
        print("="*70)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print("✓ Model loaded successfully\n")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        # Create model with same config as training
        audio_config = AudioBranchConfig(
            feature_dim=self.config["audio_dim"],
            num_layers=self.config["num_audio_layers"],
            num_segments=8,
        )
        
        visual_config = VisualBranchConfig(
            use_pretrained_encoder=False,
            feature_dim=self.config["visual_dim"],
            temporal_depth=self.config["num_visual_layers"],
        )
        
        fusion_config = LFM2FusionConfig(
            use_pretrained=False,
            num_layers=self.config["num_fusion_layers"],
            hidden_dim=self.config["fusion_hidden_dim"],
            audio_dim=self.config["audio_dim"],
            visual_dim=self.config["visual_dim"],
            output_dim=self.config["fusion_output_dim"],
        )
        
        model = MultimodalFER(
            audio_config=audio_config,
            visual_config=visual_config,
            fusion_config=fusion_config,
            num_classes=8,
            num_segments=8,
        )
        
        # Load checkpoint
        checkpoint = torch.load(
            self.config["checkpoint_path"],
            map_location=self.device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move to device
        model = model.to(self.device)
        
        # Print checkpoint info
        print(f"\nCheckpoint Info:")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
        print(f"  Best UAR: {checkpoint['metrics']['uar']:.4f}")
        print(f"  Accuracy: {checkpoint['metrics']['accuracy']:.4f}")
        
        return model
    
    def extract_audio(self, video_path: str) -> torch.Tensor:
        """
        Extract audio from video.
        For now, returns silence (you can add ffmpeg extraction later).
        """
        # Return silence (3 seconds at 16kHz)
        audio = torch.zeros(48000)
        return audio
    
    def extract_video(self, video_path: str) -> torch.Tensor:
        """Extract video frames."""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Sample 16 frames uniformly
        num_frames = 16
        if len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < num_frames:
            # Repeat last frame
            while len(frames) < num_frames:
                frames.append(frames[-1])
        
        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        return frames
    
    @torch.no_grad()
    def predict(self, video_path: str) -> Dict:
        """Predict emotion from video."""
        print(f"Processing: {video_path}\n")
        
        # Extract features
        print("[1/3] Extracting audio...")
        audio = self.extract_audio(video_path)
        
        print("[2/3] Extracting video frames...")
        video = self.extract_video(video_path)
        
        # Add batch dimension
        audio = audio.unsqueeze(0).to(self.device)
        video = video.unsqueeze(0).to(self.device)
        
        # Predict
        print("[3/3] Running inference...\n")
        outputs = self.model(audio, video)
        
        # Get results
        probabilities = outputs["probabilities"][0]  # [8]
        prediction_idx = probabilities.argmax().item()
        confidence = probabilities[prediction_idx].item()
        
        # Get top-K
        top_k = self.config["show_top_k"]
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        result = {
            "predicted_emotion": self.emotion_labels[prediction_idx],
            "predicted_idx": prediction_idx,
            "confidence": confidence,
            "top_k": [
                {
                    "emotion": self.emotion_labels[idx],
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
            "all_probabilities": {
                emotion: prob.item()
                for emotion, prob in zip(self.emotion_labels, probabilities)
            }
        }
        
        return result
    
    def print_result(self, result: Dict):
        """Print prediction results."""
        print("="*70)
        print("PREDICTION RESULT")
        print("="*70)
        
        # Main prediction
        print(f"\n🎭 Predicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Top-K predictions
        print(f"\n📊 Top-{self.config['show_top_k']} Predictions:")
        for i, pred in enumerate(result['top_k'], 1):
            bar_length = int(pred['probability'] * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {i}. {pred['emotion']:<10} {bar} {pred['probability']:.2%}")
        
        # All probabilities
        if self.config["show_all_probs"]:
            print(f"\n📈 All Probabilities:")
            for emotion, prob in result["all_probabilities"].items():
                bar_length = int(prob * 30)
                bar = "▓" * bar_length + "░" * (30 - bar_length)
                print(f"   {emotion:<10} {bar} {prob:.4f}")
        
        print("="*70)


def main():
    """Main inference function."""
    # Create predictor
    predictor = EmotionPredictor(CONFIG)
    
    # Check video exists
    video_path = CONFIG["video_path"]
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        print("\nAvailable test videos:")
        test_dir = Path("data/test_samples")
        if test_dir.exists():
            for video in test_dir.glob("*.mp4"):
                print(f"  - {video}")
        return
    
    # Predict
    result = predictor.predict(video_path)
    
    # Print result
    predictor.print_result(result)
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()
