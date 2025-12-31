"""
Inference Script for Single Video

Load model checkpoint and predict emotion from a single video file.
"""

import torch
import torch.nn as nn
import torchaudio
import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
import subprocess
import tempfile
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER


class VideoEmotionPredictor:
    """Predict emotion from video file."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        audio_sample_rate: int = 16000,
        audio_duration: float = 3.0,
        num_video_frames: int = 16,
        video_size: Tuple[int, int] = (224, 224),
    ):
        self.device = device
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        
        # Emotion labels
        self.emotion_labels = [
            "neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"
        ]
        
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        # Create model
        model = MultimodalFER(num_classes=8, num_segments=8)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def extract_audio(self, video_path: str) -> torch.Tensor:
        """
        Extract audio from video file.
        
        Returns:
            audio: [T] - Audio waveform at 16kHz
        """
        try:
            # Extract audio to temporary file using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',  # Mono
                '-y', tmp_path,
                '-loglevel', 'quiet'
            ]
            
            subprocess.run(cmd, check=True)
            
            # Load audio
            waveform, sr = torchaudio.load(tmp_path)
            
            # Clean up
            Path(tmp_path).unlink()
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"Warning: Failed to extract audio: {e}")
            print("Using silence as audio input")
            waveform = torch.zeros(int(self.audio_sample_rate * self.audio_duration))
        
        # Pad or trim to fixed duration
        target_length = int(self.audio_sample_rate * self.audio_duration)
        
        if waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:target_length]
        
        return waveform
    
    def extract_video(self, video_path: str) -> torch.Tensor:
        """
        Extract video frames.
        
        Returns:
            frames: [T, C, H, W] - Video frames
        """
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, self.video_size)
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            print("Warning: No frames extracted, using blank frames")
            frames = [np.zeros((*self.video_size, 3), dtype=np.uint8) 
                     for _ in range(self.num_video_frames)]
        
        # Sample frames uniformly
        if len(frames) > self.num_video_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_video_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_video_frames:
            # Repeat last frame
            while len(frames) < self.num_video_frames:
                frames.append(frames[-1])
        
        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        return frames
    
    @torch.no_grad()
    def predict(self, video_path: str, return_all_probs: bool = False):
        """
        Predict emotion from video file.
        
        Args:
            video_path: Path to video file
            return_all_probs: Return all class probabilities
            
        Returns:
            prediction: Dict with prediction results
        """
        print(f"\nProcessing video: {video_path}")
        
        # Extract audio and video
        print("  [1/3] Extracting audio...")
        audio = self.extract_audio(video_path)
        
        print("  [2/3] Extracting video frames...")
        video = self.extract_video(video_path)
        
        # Add batch dimension
        audio = audio.unsqueeze(0).to(self.device)  # [1, T]
        video = video.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        
        # Predict
        print("  [3/3] Running inference...")
        outputs = self.model(audio, video)
        
        # Get results
        probabilities = outputs["probabilities"][0]  # [8]
        prediction_idx = probabilities.argmax().item()
        confidence = probabilities[prediction_idx].item()
        
        # Get top-3
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        
        result = {
            "predicted_emotion": self.emotion_labels[prediction_idx],
            "predicted_idx": prediction_idx,
            "confidence": confidence,
            "top3": [
                {
                    "emotion": self.emotion_labels[idx],
                    "probability": prob.item()
                }
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        }
        
        if return_all_probs:
            result["all_probabilities"] = {
                emotion: prob.item()
                for emotion, prob in zip(self.emotion_labels, probabilities)
            }
        
        return result
    
    def print_prediction(self, result: dict):
        """Print prediction results in a nice format."""
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        
        print(f"\n🎭 Predicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        print(f"\n📊 Top-3 Predictions:")
        for i, pred in enumerate(result['top3'], 1):
            bar_length = int(pred['probability'] * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {i}. {pred['emotion']:<10} {bar} {pred['probability']:.2%}")
        
        if "all_probabilities" in result:
            print(f"\n📈 All Probabilities:")
            for emotion, prob in result["all_probabilities"].items():
                print(f"   {emotion:<10}: {prob:.4f}")
        
        print("="*60)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Predict emotion from video")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--show-all-probs",
        action="store_true",
        help="Show all class probabilities"
    )
    
    args = parser.parse_args()
    
    # Check video file exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    print("\n" + "="*60)
    print("EMOTION RECOGNITION - INFERENCE")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Create predictor
    predictor = VideoEmotionPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Predict
    result = predictor.predict(
        video_path=args.video,
        return_all_probs=args.show_all_probs,
    )
    
    # Print result
    predictor.print_prediction(result)


if __name__ == "__main__":
    main()
