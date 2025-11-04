#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Prepare dummy data for testing AR Transformer training

import os
import json
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def create_dummy_data(output_dir, num_samples=10, min_duration=2.0, max_duration=8.0):
    """Create dummy audio-motion pairs for testing.

    Args:
        output_dir: Output directory for dummy data
        num_samples: Number of samples to create
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'audios'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'motions'), exist_ok=True)

    metadata = []
    sample_rate = 16000
    motion_fps = 25

    print(f"Creating {num_samples} dummy samples...")

    for i in tqdm(range(num_samples)):
        # Random duration
        duration = np.random.uniform(min_duration, max_duration)

        # Create dummy audio (white noise)
        num_audio_samples = int(duration * sample_rate)
        audio = torch.randn(1, num_audio_samples) * 0.1

        # Save audio
        audio_filename = f'sample_{i:04d}.wav'
        audio_path = os.path.join(output_dir, 'audios', audio_filename)
        torchaudio.save(audio_path, audio, sample_rate)

        # Create dummy motion (random walk to make it more realistic)
        num_frames = int(duration * motion_fps)
        motion = np.zeros((num_frames, 106), dtype=np.float32)

        # Initialize with small random values
        motion[0] = np.random.randn(106) * 0.1

        # Random walk for smooth motion
        for t in range(1, num_frames):
            motion[t] = motion[t-1] + np.random.randn(106) * 0.05
            # Clip to prevent explosion
            motion[t] = np.clip(motion[t], -2.0, 2.0)

        # Save motion
        motion_filename = f'sample_{i:04d}.npy'
        motion_path = os.path.join(output_dir, 'motions', motion_filename)
        np.save(motion_path, motion)

        # Add to metadata
        metadata.append({
            'audio_path': f'audios/{audio_filename}',
            'motion_path': f'motions/{motion_filename}',
            'duration': float(duration)
        })

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDummy data created successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"Metadata file: {metadata_path}")

    # Print statistics
    total_duration = sum(item['duration'] for item in metadata)
    avg_duration = total_duration / num_samples
    print(f"\nStatistics:")
    print(f"  Total duration: {total_duration:.2f} seconds")
    print(f"  Average duration: {avg_duration:.2f} seconds")
    print(f"  Total frames: {sum(int(item['duration'] * motion_fps) for item in metadata)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dummy data for AR Transformer training')
    parser.add_argument('--output_dir', type=str, default='dummy_data/train',
                        help='Output directory for dummy data')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to create')
    parser.add_argument('--min_duration', type=float, default=2.0,
                        help='Minimum duration in seconds')
    parser.add_argument('--max_duration', type=float, default=8.0,
                        help='Maximum duration in seconds')

    args = parser.parse_args()

    create_dummy_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )


if __name__ == '__main__':
    main()
