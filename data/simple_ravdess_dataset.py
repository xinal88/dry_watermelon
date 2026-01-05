"""
Simple RAVDESS Dataset Loader - Works with ANY folder structure

Just finds all .mp4 files and parses filenames.
No complex folder structure required!
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import warnings


class SimpleRAVDESSDataset(Dataset):
    """
    Simple RAVDESS Dataset - works with any folder structure.
    
    Just finds all .mp4 files recursively and parses filenames.
    
    Filename format: XX-YY-ZZ-AA-BB-CC-DD.mp4
    - XX: Modality (01=audio-video, 02=video-only, 03=audio-only)
    - YY: Vocal channel (01=speech, 02=song)
    - ZZ: Emotion (01-08)
    - AA: Emotional intensity (01=normal, 02=strong)
    - BB: Statement (01 or 02)
    - CC: Repetition (01 or 02)
    - DD: Actor (01-24)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modality: str = "speech",
        audio_sample_rate: int = 16000,
        audio_duration: float = 3.0,
        num_video_frames: int = 16,
        video_size: Tuple[int, int] = (224, 224),
        use_audio: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.use_audio = use_audio
        
        # Emotion mapping
        self.emotion_map = {
            "01": 0, "02": 1, "03": 2, "04": 3,
            "05": 4, "06": 5, "07": 6, "08": 7,
        }
        
        self.emotion_names = [
            "neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"
        ]
        
        # Load files
        self.video_files = self._load_files()
        
        print(f"Loaded {len(self.video_files)} videos for {split} split ({modality})")
    
    def _parse_filename(self, filename: str) -> Dict:
        """Parse RAVDESS filename."""
        try:
            parts = filename.replace('.mp4', '').split('-')
            if len(parts) != 7:
                return None
            
            return {
                "modality_code": parts[0],
                "vocal_channel": parts[1],  # 01=speech, 02=song
                "emotion": parts[2],
                "intensity": parts[3],
                "statement": parts[4],
                "repetition": parts[5],
                "actor": int(parts[6]),  # Actor number as int
            }
        except:
            return None
    
    def _load_files(self) -> List[Path]:
        """Load all .mp4 files and filter by split and modality."""
        # Find ALL .mp4 files recursively
        all_videos = list(self.data_dir.rglob("*.mp4"))
        
        # Filter by modality and split
        filtered_videos = []
        
        for video_path in all_videos:
            metadata = self._parse_filename(video_path.name)
            
            if metadata is None:
                continue
            
            # Filter by modality (speech=01, song=02)
            if self.modality == "speech" and metadata["vocal_channel"] != "01":
                continue
            if self.modality == "song" and metadata["vocal_channel"] != "02":
                continue
            
            # Filter by split (based on actor number)
            actor_num = metadata["actor"]
            
            if self.split == "train" and 1 <= actor_num <= 16:
                filtered_videos.append(video_path)
            elif self.split == "val" and 17 <= actor_num <= 20:
                filtered_videos.append(video_path)
            elif self.split == "test" and 21 <= actor_num <= 24:
                filtered_videos.append(video_path)
        
        return sorted(filtered_videos)
    
    def _extract_audio(self, video_path: Path) -> torch.Tensor:
        """Extract audio from video."""
        if not self.use_audio:
            target_length = int(self.audio_sample_rate * self.audio_duration)
            return torch.zeros(target_length)
        
        try:
            import torchaudio
            import subprocess
            import tempfile
            
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
            waveform, sr = torchaudio.load(tmp_path)
            Path(tmp_path).unlink()
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            
        except Exception as e:
            warnings.warn(f"Failed to extract audio from {video_path}: {e}")
            target_length = int(self.audio_sample_rate * self.audio_duration)
            return torch.zeros(target_length)
        
        # Pad or trim
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
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            while len(frames) < self.num_video_frames:
                frames.append(frames[-1])
        
        # Convert to tensor
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames = frames / 255.0
        
        return frames
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        video_path = self.video_files[idx]
        
        # Parse filename
        metadata = self._parse_filename(video_path.name)
        
        # Get emotion label
        emotion_code = metadata["emotion"]
        label = self.emotion_map[emotion_code]
        
        # Extract audio and video
        audio = self._extract_audio(video_path)
        video = self._extract_video(video_path)
        
        metadata["filename"] = video_path.name
        
        return audio, video, label, metadata


def create_simple_ravdess_dataloaders(
    data_dir: str,
    modality: str = "speech",
    batch_size: int = 8,
    num_workers: int = 4,
    use_audio: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders using simple dataset loader.
    
    Works with ANY folder structure - just finds all .mp4 files!
    """
    train_dataset = SimpleRAVDESSDataset(
        data_dir=data_dir,
        split="train",
        modality=modality,
        use_audio=use_audio,
    )
    
    val_dataset = SimpleRAVDESSDataset(
        data_dir=data_dir,
        split="val",
        modality=modality,
        use_audio=use_audio,
    )
    
    test_dataset = SimpleRAVDESSDataset(
        data_dir=data_dir,
        split="test",
        modality=modality,
        use_audio=use_audio,
    )
    
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
