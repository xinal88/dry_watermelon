"""
Test Dataset Loader for RAVDESS samples

Loads video samples from data/test_samples/ for testing pipeline.
RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4
Example: 01-02-01-01-01-01-01.mp4
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings


class RAVDESSTestDataset(Dataset):
    """
    Test dataset for RAVDESS video samples.
    
    RAVDESS filename format:
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4
    
    Emotion codes:
    01 = neutral, 02 = calm, 03 = happy, 04 = sad,
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    
    def __init__(
        self,
        data_dir: str = "data/test_samples",
        audio_sample_rate: int = 16000,
        audio_duration: float = 3.0,
        num_video_frames: int = 16,
        video_size: Tuple[int, int] = (224, 224),
    ):
        self.data_dir = Path(data_dir)
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        
        # Find all video files
        self.video_files = sorted(list(self.data_dir.glob("*.mp4")))
        
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {data_dir}")
        
        print(f"Found {len(self.video_files)} video samples")
        
        # Emotion mapping (RAVDESS codes to indices)
        self.emotion_map = {
            "01": 0,  # neutral
            "02": 1,  # calm
            "03": 2,  # happy
            "04": 3,  # sad
            "05": 4,  # angry
            "06": 5,  # fearful
            "07": 6,  # disgust
            "08": 7,  # surprised
        }
        
        self.emotion_names = [
            "neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"
        ]
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse RAVDESS filename.
        
        Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4
        Example: 01-02-01-01-01-01-01.mp4
        """
        parts = filename.stem.split('-')
        
        if len(parts) != 7:
            warnings.warn(f"Unexpected filename format: {filename}")
            return {"emotion_code": "01", "emotion_idx": 0}
        
        emotion_code = parts[2]
        emotion_idx = self.emotion_map.get(emotion_code, 0)
        
        return {
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion_code": emotion_code,
            "emotion_idx": emotion_idx,
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
            "actor": parts[6],
        }
    
    def load_audio(self, video_path: Path) -> torch.Tensor:
        """
        Extract audio from video file.
        
        Returns:
            audio: [T] - Audio waveform at 16kHz
        """
        try:
            # Try using torchaudio to extract audio
            import subprocess
            import tempfile
            
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', str(video_path),
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
            warnings.warn(f"Failed to extract audio from {video_path}: {e}")
            # Return silence
            waveform = torch.zeros(int(self.audio_sample_rate * self.audio_duration))
        
        # Pad or trim to fixed duration
        target_length = int(self.audio_sample_rate * self.audio_duration)
        
        if waveform.shape[0] < target_length:
            # Pad
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Trim
            waveform = waveform[:target_length]
        
        return waveform
    
    def load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load video frames.
        
        Returns:
            frames: [T, C, H, W] - Video frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
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
            warnings.warn(f"No frames extracted from {video_path}")
            # Return blank frames
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
        frames = np.stack(frames)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, C, H, W]
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Get a sample.
        
        Returns:
            audio: [T_audio] - Audio waveform
            video: [T, C, H, W] - Video frames
            label: int - Emotion label
            metadata: Dict - Sample metadata
        """
        video_path = self.video_files[idx]
        
        # Parse filename
        metadata = self.parse_filename(video_path)
        metadata["filename"] = video_path.name
        
        # Load audio
        audio = self.load_audio(video_path)
        
        # Load video
        video = self.load_video(video_path)
        
        # Get label
        label = metadata["emotion_idx"]
        
        return audio, video, label, metadata


def create_test_dataloader(
    data_dir: str = "data/test_samples",
    batch_size: int = 2,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        data_dir: Directory with test samples
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        DataLoader
    """
    dataset = RAVDESSTestDataset(data_dir=data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    """Test the dataset loader."""
    print("Testing RAVDESS Test Dataset Loader")
    print("="*60)
    
    # Create dataset
    dataset = RAVDESSTestDataset()
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load first sample
    print("\nLoading first sample...")
    audio, video, label, metadata = dataset[0]
    
    print(f"\nSample metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nData shapes:")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")
    print(f"  Label: {label} ({dataset.emotion_names[label]})")
    
    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = create_test_dataloader(batch_size=2)
    
    for batch_idx, (audio_batch, video_batch, labels_batch, metadata_batch) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Audio: {audio_batch.shape}")
        print(f"  Video: {video_batch.shape}")
        print(f"  Labels: {labels_batch}")
        break
    
    print("\n✓ Dataset loader test passed!")
