"""
RAVDESS Dataset Loader

Full RAVDESS dataset with train/val/test splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import warnings


class RAVDESSDataset(Dataset):
    """
    RAVDESS Multimodal Dataset.
    
    File naming: XX-YY-ZZ-AA-BB-CC-DD.mp4
    - XX: Modality (01=audio-video, 02=video-only, 03=audio-only)
    - YY: Vocal channel (01=speech, 02=song)
    - ZZ: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    - AA: Emotional intensity (01=normal, 02=strong)
    - BB: Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    - CC: Repetition (01=1st, 02=2nd)
    - DD: Actor (01-24, odd=male, even=female)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modality: str = "speech",  # "speech" or "song"
        audio_sample_rate: int = 16000,
        audio_duration: float = 3.0,
        num_video_frames: int = 16,
        video_size: Tuple[int, int] = (224, 224),
        use_audio: bool = True,
    ):
        """
        Args:
            data_dir: Path to RAVDESS data directory
            split: "train", "val", or "test"
            modality: "speech" or "song"
            audio_sample_rate: Audio sampling rate
            audio_duration: Audio duration in seconds
            num_video_frames: Number of frames to extract
            video_size: Video frame size (H, W)
            use_audio: Whether to extract audio (requires ffmpeg)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.use_audio = use_audio
        
        # Emotion mapping (RAVDESS code -> index)
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
        
        # Load file list
        self.video_files = self._load_files()
        
        print(f"Loaded {len(self.video_files)} videos for {split} split ({modality})")
    
    def _load_files(self) -> List[Path]:
        """Load video files based on split and modality."""
        # Determine which folders to use
        if self.modality == "speech":
            pattern = "Video_Speech_Actor_*"
        else:  # song
            pattern = "Video_Song_Actor_*"
        
        # Get all actor folders
        actor_folders = sorted(self.data_dir.glob(pattern))
        
        # Split actors: train (1-16), val (17-20), test (21-24)
        if self.split == "train":
            actor_range = range(1, 17)  # Actors 01-16
        elif self.split == "val":
            actor_range = range(17, 21)  # Actors 17-20
        else:  # test
            actor_range = range(21, 25)  # Actors 21-24
        
        # Collect video files
        video_files = []
        for folder in actor_folders:
            # Extract actor number from folder name
            actor_num = int(folder.name.split("_")[-1])
            
            if actor_num in actor_range:
                # Get all .mp4 files in this folder
                videos = list(folder.glob("*.mp4"))
                video_files.extend(videos)
        
        return sorted(video_files)
    
    def _parse_filename(self, filename: str) -> Dict:
        """Parse RAVDESS filename to extract metadata."""
        parts = filename.stem.split("-")
        
        return {
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion": parts[2],
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
            "actor": parts[6],
        }
    
    def _extract_audio(self, video_path: Path) -> torch.Tensor:
        """
        Extract audio from video.
        Returns silence if extraction fails (ffmpeg not available).
        """
        if not self.use_audio:
            # Return silence
            target_length = int(self.audio_sample_rate * self.audio_duration)
            return torch.zeros(target_length)
        
        try:
            import torchaudio
            import subprocess
            import tempfile
            
            # Extract audio using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',
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
            
            waveform = waveform.squeeze(0)
            
        except Exception as e:
            # Fallback to silence
            warnings.warn(f"Failed to extract audio from {video_path}: {e}")
            target_length = int(self.audio_sample_rate * self.audio_duration)
            return torch.zeros(target_length)
        
        # Pad or trim to fixed duration
        target_length = int(self.audio_sample_rate * self.audio_duration)
        
        if waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:target_length]
        
        return waveform
    
    def _extract_video(self, video_path: Path) -> torch.Tensor:
        """Extract video frames."""
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
            raise ValueError(f"No frames extracted from {video_path}")
        
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
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Returns:
            audio: [T_audio] - Audio waveform
            video: [T, C, H, W] - Video frames
            label: int - Emotion label (0-7)
            metadata: dict - File metadata
        """
        video_path = self.video_files[idx]
        
        # Parse filename
        metadata = self._parse_filename(video_path)
        
        # Get emotion label
        emotion_code = metadata["emotion"]
        label = self.emotion_map[emotion_code]
        
        # Extract audio and video
        audio = self._extract_audio(video_path)
        video = self._extract_video(video_path)
        
        # Add filename to metadata
        metadata["filename"] = video_path.name
        
        return audio, video, label, metadata


def create_ravdess_dataloaders(
    data_dir: str,
    modality: str = "speech",
    batch_size: int = 8,
    num_workers: int = 4,
    use_audio: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders for RAVDESS.
    
    Args:
        data_dir: Path to RAVDESS data directory
        modality: "speech" or "song"
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_audio: Whether to extract audio (requires ffmpeg)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = RAVDESSDataset(
        data_dir=data_dir,
        split="train",
        modality=modality,
        use_audio=use_audio,
    )
    
    val_dataset = RAVDESSDataset(
        data_dir=data_dir,
        split="val",
        modality=modality,
        use_audio=use_audio,
    )
    
    test_dataset = RAVDESSDataset(
        data_dir=data_dir,
        split="test",
        modality=modality,
        use_audio=use_audio,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
