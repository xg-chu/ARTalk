#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Dataset for AR Transformer Training

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset


class AudioMotionDataset(Dataset):
    """Dataset for loading audio-motion pairs for AR Transformer training.

    Expected data format:
    data_root/
        metadata.json  # List of samples with audio_path, motion_path, duration
        audios/        # Audio files (.wav, 16kHz mono)
        motions/       # Motion files (.npy, shape: [T, 106])

    metadata.json format:
    [
        {
            "audio_path": "audios/sample1.wav",
            "motion_path": "motions/sample1.npy",
            "duration": 5.2  # seconds (optional)
        },
        ...
    ]
    """

    def __init__(
        self,
        data_root,
        metadata_file="metadata.json",
        sample_rate=16000,
        motion_fps=25,
        chunk_duration=4.0,  # seconds (matches 100 frames at 25fps)
        overlap=0.0,  # overlap between chunks (seconds)
        min_chunk_duration=1.0,  # minimum chunk duration
        augment=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.motion_fps = motion_fps
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.min_chunk_duration = min_chunk_duration
        self.augment = augment

        # Load metadata
        metadata_path = os.path.join(data_root, metadata_file)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Create chunks from samples
        self.chunks = self._create_chunks()
        print(f"Loaded {len(self.metadata)} samples, created {len(self.chunks)} chunks")

    def _create_chunks(self):
        """Create training chunks from full sequences."""
        chunks = []
        chunk_frames = int(self.chunk_duration * self.motion_fps)
        overlap_frames = int(self.overlap * self.motion_fps)
        stride_frames = chunk_frames - overlap_frames
        min_frames = int(self.min_chunk_duration * self.motion_fps)

        for sample_idx, sample in enumerate(self.metadata):
            # Load motion to get actual length
            motion_path = os.path.join(self.data_root, sample['motion_path'])
            motion = np.load(motion_path)
            total_frames = motion.shape[0]

            # Create chunks with sliding window
            start = 0
            while start < total_frames:
                end = min(start + chunk_frames, total_frames)
                if end - start >= min_frames:
                    chunks.append({
                        'sample_idx': sample_idx,
                        'start_frame': start,
                        'end_frame': end,
                        'audio_path': sample['audio_path'],
                        'motion_path': sample['motion_path']
                    })
                if end >= total_frames:
                    break
                start += stride_frames

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        # Load audio
        audio_path = os.path.join(self.data_root, chunk['audio_path'])
        audio, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)  # [T]

        # Load motion
        motion_path = os.path.join(self.data_root, chunk['motion_path'])
        motion = np.load(motion_path)  # [T, 106]
        motion = torch.from_numpy(motion).float()

        # Extract chunk
        start_frame = chunk['start_frame']
        end_frame = chunk['end_frame']
        motion_chunk = motion[start_frame:end_frame]

        # Extract corresponding audio chunk
        start_sample = int(start_frame / self.motion_fps * self.sample_rate)
        end_sample = int(end_frame / self.motion_fps * self.sample_rate)
        audio_chunk = audio[start_sample:end_sample]

        # Pad to fixed length (100 frames = 4 seconds)
        target_frames = 100
        target_audio_samples = int(target_frames / self.motion_fps * self.sample_rate)

        # Pad motion
        if motion_chunk.shape[0] < target_frames:
            pad_frames = target_frames - motion_chunk.shape[0]
            motion_chunk = torch.cat([
                motion_chunk,
                torch.zeros(pad_frames, motion_chunk.shape[1])
            ], dim=0)
        elif motion_chunk.shape[0] > target_frames:
            motion_chunk = motion_chunk[:target_frames]

        # Pad audio
        if audio_chunk.shape[0] < target_audio_samples:
            pad_samples = target_audio_samples - audio_chunk.shape[0]
            audio_chunk = torch.cat([
                audio_chunk,
                torch.zeros(pad_samples)
            ], dim=0)
        elif audio_chunk.shape[0] > target_audio_samples:
            audio_chunk = audio_chunk[:target_audio_samples]

        # Apply augmentation if enabled
        if self.augment and torch.rand(1).item() < 0.5:
            # Random volume scaling
            volume_scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            audio_chunk = audio_chunk * volume_scale

        return {
            'audio': audio_chunk,  # [audio_samples]
            'motion': motion_chunk,  # [100, 106]
            'seq_length': end_frame - start_frame,  # actual length before padding
        }


class AudioMotionCollateFn:
    """Collate function for batching audio-motion pairs."""

    def __init__(self, sample_rate=16000, motion_fps=25):
        self.sample_rate = sample_rate
        self.motion_fps = motion_fps

    def __call__(self, batch):
        audios = torch.stack([item['audio'] for item in batch])
        motions = torch.stack([item['motion'] for item in batch])
        seq_lengths = torch.tensor([item['seq_length'] for item in batch])

        return {
            'audio': audios,  # [B, audio_samples]
            'motion': motions,  # [B, 100, 106]
            'seq_length': seq_lengths,  # [B]
        }


# Example usage and testing
if __name__ == '__main__':
    # Create dummy data for testing
    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'audios'))
    os.makedirs(os.path.join(temp_dir, 'motions'))

    # Create dummy samples
    metadata = []
    for i in range(5):
        # Create dummy audio (5 seconds at 16kHz)
        audio = torch.randn(1, 16000 * 5)
        audio_path = f'audios/sample{i}.wav'
        torchaudio.save(
            os.path.join(temp_dir, audio_path),
            audio, 16000
        )

        # Create dummy motion (125 frames = 5 seconds at 25fps)
        motion = np.random.randn(125, 106).astype(np.float32)
        motion_path = f'motions/sample{i}.npy'
        np.save(os.path.join(temp_dir, motion_path), motion)

        metadata.append({
            'audio_path': audio_path,
            'motion_path': motion_path,
            'duration': 5.0
        })

    # Save metadata
    with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Test dataset
    dataset = AudioMotionDataset(
        data_root=temp_dir,
        chunk_duration=4.0,
        overlap=0.0
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Motion shape: {sample['motion'].shape}")
    print(f"Sequence length: {sample['seq_length']}")

    # Test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=AudioMotionCollateFn(),
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"\nBatch audio shape: {batch['audio'].shape}")
    print(f"Batch motion shape: {batch['motion'].shape}")
    print(f"Batch seq_length shape: {batch['seq_length'].shape}")

    # Clean up
    shutil.rmtree(temp_dir)
    print("\nDataset test passed!")
