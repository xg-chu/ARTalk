# AR Transformer Training Guide

This document explains how to train the AR Transformer model for speech-driven facial animation.

## Overview

The AR Transformer is trained to predict quantized motion codes from audio input in an autoregressive manner. The model generates motion hierarchically across 5 patch levels: [1, 5, 25, 50, 100] frames.

## Architecture Summary

### Model Components

1. **Bitwise VAE** (Frozen during AR training)
   - Encodes motion (106-dim) → quantized codes (32-dim binary)
   - Pre-trained separately on motion data
   - Used to generate target codes for AR training

2. **Audio Encoder** (Frozen)
   - Wav2Vec2-XLS-R-300M (default): 1024-dim features
   - Alternative: MIMI (512-dim features)
   - Extracts speech representations from audio

3. **Style Encoder** (Trainable)
   - Input: 50 frames (2 seconds) of reference motion
   - Output: 128-dim style embedding
   - Provides style conditioning for diverse motion generation

4. **AR Transformer** (Trainable)
   - 12 layers, 768-dim embeddings, 12 attention heads
   - Adaptive LayerNorm conditioned on audio
   - Predicts binary codes hierarchically

### Training Process

- **Forward Pass**: Teacher forcing with ground truth codes
- **Loss**: Cross-entropy loss for binary code prediction
- **Optimization**: AdamW with cosine annealing schedule
- **Batch Size**: 8 (default, adjust based on GPU memory)

## Data Preparation

### Data Format

Your training data should be organized as follows:

```
data_root/
├── metadata.json         # List of audio-motion pairs
├── audios/              # Audio files (.wav, 16kHz mono)
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── motions/             # Motion files (.npy, shape: [T, 106])
    ├── sample1.npy
    ├── sample2.npy
    └── ...
```

### Metadata Format

Create a `metadata.json` file with the following structure:

```json
[
    {
        "audio_path": "audios/sample1.wav",
        "motion_path": "motions/sample1.npy",
        "duration": 5.2
    },
    {
        "audio_path": "audios/sample2.wav",
        "motion_path": "motions/sample2.npy",
        "duration": 8.1
    }
]
```

### Motion Parameters

Motion parameters should be a NumPy array of shape `[T, 106]`:
- First 100 dimensions: FLAME expression codes
- Last 6 dimensions: FLAME pose codes (global rotation + jaw/neck/eyes)
- Temporal dimension T: varies by sample (25 fps)

### Audio Requirements

- Sample rate: 16kHz
- Channels: Mono
- Format: WAV
- Duration: Should match motion duration (T frames / 25 fps)

## Training

### Basic Training Command

```bash
python train.py \
    --data_root /path/to/training_data \
    --config assets/config.json \
    --audio_encoder wav2vec \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --output_dir outputs/ar_training
```

### Advanced Options

```bash
python train.py \
    --data_root /path/to/training_data \
    --val_data_root /path/to/validation_data \
    --config assets/config.json \
    --audio_encoder wav2vec \
    --pretrained_vae /path/to/vae_checkpoint.pth \
    --freeze_vae \
    --freeze_audio_encoder \
    --batch_size 8 \
    --num_workers 4 \
    --num_epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --lr_schedule cosine \
    --label_smoothing 0.1 \
    --recon_loss_weight 0.0 \
    --output_dir outputs/ar_training \
    --log_interval 10 \
    --save_interval 5 \
    --eval_interval 1 \
    --amp
```

### Training Arguments

#### Data Arguments
- `--data_root`: Root directory of training data (required)
- `--metadata_file`: Metadata file name (default: metadata.json)
- `--val_data_root`: Root directory of validation data (optional)

#### Model Arguments
- `--config`: Model config file (default: assets/config.json)
- `--audio_encoder`: Audio encoder type (wav2vec or mimi)
- `--pretrained_vae`: Path to pretrained VAE checkpoint
- `--pretrained_model`: Path to pretrained AR model (for resume)
- `--freeze_vae`: Freeze VAE during training (recommended)
- `--freeze_audio_encoder`: Freeze audio encoder (recommended)

#### Training Arguments
- `--batch_size`: Batch size (default: 8)
- `--num_workers`: Number of data loading workers (default: 4)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_epochs`: Number of warmup epochs (default: 5)
- `--lr_schedule`: Learning rate schedule (cosine/step/constant)

#### Loss Arguments
- `--label_smoothing`: Label smoothing for cross-entropy (default: 0.1)
- `--recon_loss_weight`: Weight for motion reconstruction loss (default: 0.0)

#### Logging Arguments
- `--output_dir`: Output directory (default: outputs/ar_training)
- `--log_interval`: Log interval in steps (default: 10)
- `--save_interval`: Save interval in epochs (default: 1)
- `--eval_interval`: Evaluation interval in epochs (default: 1)

#### Misc Arguments
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (default: cuda)
- `--amp`: Use automatic mixed precision

## Monitoring Training

### TensorBoard

Training logs are saved to `{output_dir}/logs/`. Monitor training with:

```bash
tensorboard --logdir outputs/ar_training/logs
```

### Logged Metrics

- `train/loss`: Total training loss
- `train/accuracy`: Binary code prediction accuracy
- `train/ar_loss`: AR cross-entropy loss
- `train/lr`: Learning rate
- `train/level_{i}_loss`: Loss for each patch level
- `train/level_{i}_acc`: Accuracy for each patch level
- `val/loss`: Validation loss
- `val/accuracy`: Validation accuracy

## Checkpoints

Checkpoints are saved to `{output_dir}/`:
- `checkpoint_epoch_{epoch}.pth`: Regular checkpoint
- `best_model.pth`: Best model based on validation loss

### Checkpoint Contents

- `epoch`: Current epoch
- `global_step`: Global training step
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `args`: Training arguments

### Loading Checkpoint

```python
checkpoint = torch.load('outputs/ar_training/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Inference with Trained Model

Update `inference.py` to load your trained model:

```python
# In inference.py, modify the model loading:
checkpoint = torch.load('outputs/ar_training/best_model.pth')
self.model.load_state_dict(checkpoint['model_state_dict'])
```

Or use the provided checkpoint path in the ARTAvatarInferEngine initialization.

## Tips for Training

### GPU Memory

- Default batch size: 8 (requires ~16GB VRAM)
- Reduce batch size if out of memory
- Enable AMP (`--amp`) for reduced memory usage

### Training Time

- ~1 hour per epoch on RTX 3090 (depends on dataset size)
- Recommended: 50-100 epochs for good results

### Data Augmentation

The dataset applies random volume scaling to audio (0.8-1.2x). Additional augmentations can be added in `dataset.py`.

### Hyperparameters

- **Learning rate**: 1e-4 works well, can try 5e-5 or 2e-4
- **Label smoothing**: 0.1 helps prevent overconfidence
- **Warmup**: 5 epochs helps stabilize early training
- **Weight decay**: 0.01 for regularization

### Pretrained Components

- **VAE**: Should be pre-trained separately on motion data
- **Audio Encoder**: Use pretrained Wav2Vec2/MIMI (frozen)
- **Style Encoder**: Train from scratch with AR model

## Troubleshooting

### NaN Loss

- Reduce learning rate
- Check data normalization
- Enable gradient clipping (add to optimizer)

### Poor Accuracy

- Increase training epochs
- Check data quality and alignment
- Verify audio preprocessing (16kHz, mono)
- Try reducing label smoothing

### Slow Training

- Increase batch size if possible
- Enable AMP
- Use more workers for data loading
- Profile data loading bottlenecks

## Advanced: Multi-GPU Training

For multi-GPU training, wrap the model with DataParallel or DistributedDataParallel:

```python
# Add to train.py
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

Or use `torch.distributed` for more efficient training.

## Citation

If you use this training code, please cite:

```bibtex
@article{artalk2024,
  title={ARTalk: Audio-driven Realistic Talking Head},
  author={Chu, Xuangeng and others},
  journal={arXiv preprint},
  year={2024}
}
```
