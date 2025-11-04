# AR Transformer Architecture and Training Implementation

## Overview

This document provides a comprehensive summary of the AR Transformer network architecture in ARTalk and the newly implemented training code.

## Table of Contents

1. [Architecture Details](#architecture-details)
2. [Training Implementation](#training-implementation)
3. [Key Files](#key-files)
4. [Quick Start](#quick-start)
5. [Implementation Notes](#implementation-notes)

---

## Architecture Details

### 1. High-Level Architecture

ARTalk uses an **Autoregressive Transformer** to generate speech-driven facial animations. The model predicts quantized motion codes from audio in a hierarchical, autoregressive manner.

```
Audio Input (16kHz)
    ↓
Audio Encoder (Wav2Vec2/MIMI) [Frozen]
    ↓
AR Transformer (12 layers)
    ↓
Binary Code Logits (32-dim × 5 scales)
    ↓
VAE Decoder [Frozen]
    ↓
Motion Parameters (106-dim)
    ↓
FLAME Model
    ↓
3D Facial Animation
```

### 2. Core Components

#### 2.1 Bitwise VAE (`app/modules/bitwise_vae.py`)

**Purpose**: Quantizes motion into discrete binary codes

**Architecture**:
- **Encoder**: Motion (106-dim) → Latent codes (32-dim)
  - 8 Transformer layers
  - Hidden dim: 512
  - 8 attention heads

- **Quantizer**: Multi-Scale Binary Straight-Through Quantization (BSQ)
  - 5 hierarchical scales: [1, 5, 25, 50, 100] patches
  - Binary quantization: each of 32 dimensions is 0 or 1
  - Codebook size: 2^32 (4.3 billion codes)
  - Residual learning across scales

- **Decoder**: Latent codes (32-dim) → Motion (106-dim)
  - 8 Transformer layers
  - Reconstructs motion from quantized codes

**Training**: Pre-trained separately on motion data (frozen during AR training)

#### 2.2 Audio Encoder

**Purpose**: Extract acoustic features from speech audio

**Options**:
1. **Wav2Vec2-XLS-R-300M** (Default)
   - Output: 1024-dim features
   - Pre-trained on multilingual speech
   - Operates at 50Hz

2. **MIMI** (Alternative)
   - Output: 512-dim features
   - More recent acoustic model
   - kyutai/mimi from HuggingFace

**Training**: Frozen (uses pretrained weights)

#### 2.3 Style Encoder (`app/modules/style_encoder.py`)

**Purpose**: Encode motion style from reference sequences

**Architecture**:
- Input: 50 frames (2 seconds) of reference motion (106-dim)
- 4 Transformer encoder layers
- Embed dim: 128
- 4 attention heads
- MLP ratio: 4
- Mean pooling over temporal dimension
- Output: 128-dim style embedding → projected to 768-dim

**Training**: Trained jointly with AR Transformer

#### 2.4 AR Transformer Blocks (`app/transformer.py`, `app/models.py`)

**Purpose**: Autoregressively generate motion codes conditioned on audio

**Configuration**:
```
- Depth: 12 layers
- Embed dim: 768
- Num heads: 12
- Head dim: 64
- MLP ratio: 4.0
- Drop path: 0 → 0.05 (linear schedule)
```

**Key Features**:

1. **AdaLNSelfAttn Block**:
   - Adaptive Layer Normalization conditioned on audio
   - 6 learnable parameters per token: (scale1, shift1, gamma1, scale2, shift2, gamma2)
   - Modified self-attention with optional L2 normalization
   - Feed-forward network with GELU activation
   - Residual connections with drop path regularization

2. **Modified Self-Attention**:
   - Supports L2-normalized queries and keys
   - Learnable temperature scaling
   - Custom attention bias for hierarchical masking
   - Attends to both current tokens and previous context

3. **Hierarchical Generation**:
   - Generates codes for 5 patch levels sequentially: [1, 5, 25, 50, 100]
   - Each level conditions on all previous levels
   - Hierarchical attention masking prevents attending to future/finer patches
   - Total sequence length: sum([1, 5, 25, 50, 100]) = 181 tokens

4. **Positional Embeddings**:
   - Absolute position embeddings for current generation
   - Separate position embeddings for previous context
   - Level embeddings for each of 5 hierarchical levels
   - Combined via addition

5. **Context Management**:
   - Previous context ratio (PREV_RATIO): default 1
   - Context length: 181 tokens (one chunk of 100 frames)
   - Updated at each generation step

### 3. Generation Process

#### 3.1 Inference (Autoregressive)

For each audio chunk (4 seconds = 100 frames):

1. **Encode audio** at all 5 scales via interpolation
2. **Initialize** with style condition and previous context
3. **For each patch level** (1 → 5):
   - Prepare audio condition up to current level
   - Create hierarchical attention mask
   - Add positional embeddings
   - **Run 12 AR Transformer blocks**
   - Predict logits for binary codes (32-dim × 2 classes)
   - **Argmax** to get binary codes
   - Update features for next level
4. **Decode** codes to motion via VAE decoder
5. **Update** context for next chunk

#### 3.2 Training (Teacher Forcing)

Training differs from inference:

1. **Encode ground truth motion** to get target codes via VAE encoder
2. **Use target codes** for all levels (teacher forcing)
3. **Predict logits** at each level
4. **Compute loss** between predicted logits and target codes
5. **No argmax** - use continuous logits for gradient flow

This enables parallel training across all levels within a chunk.

### 4. Motion Representation

**FLAME Parameters** (106-dim):
- **Expression codes**: 100 dimensions
  - Control facial expressions (smile, frown, etc.)
  - PCA basis learned from face scans

- **Pose codes**: 6 dimensions
  - Global rotation: 3 dims (head orientation)
  - Jaw rotation: 1 dim
  - Neck rotation: 1 dim
  - Eye rotations: 1 dim

**Normalization**:
- Mean and std computed from training dataset (ALLTALKEMICA)
- Applied in VAE encoder/decoder
- Statistics stored in `app/modules/data_stats.py`

---

## Training Implementation

### 1. New Files Created

1. **`dataset.py`** - Data loading for audio-motion pairs
2. **`losses.py`** - Loss functions for AR training
3. **`train.py`** - Main training script
4. **`prepare_dummy_data.py`** - Generate dummy data for testing
5. **`TRAINING.md`** - Comprehensive training guide
6. **`AR_TRANSFORMER_SUMMARY.md`** - This document

### 2. Modified Files

1. **`app/models.py`** - Added `forward()` method for training
2. **`assets/config.json`** - Added `AUDIO_ENCODER` field

### 3. Training Data Format

#### Directory Structure:
```
data_root/
├── metadata.json
├── audios/
│   ├── sample1.wav    # 16kHz mono
│   └── ...
└── motions/
    ├── sample1.npy    # [T, 106] numpy array
    └── ...
```

#### Metadata Format:
```json
[
    {
        "audio_path": "audios/sample1.wav",
        "motion_path": "motions/sample1.npy",
        "duration": 5.2
    }
]
```

### 4. Loss Function

**Primary Loss**: Cross-entropy for binary code prediction

```python
loss = CrossEntropy(predicted_logits, target_codes)
# predicted_logits: [B, 181, 32, 2]
# target_codes: [B, 181, 32] with values in {0, 1}
```

**Features**:
- Label smoothing (default: 0.1)
- Per-level loss weighting (optional)
- Per-level accuracy monitoring

**Optional**: Motion reconstruction loss (L1/L2) for auxiliary supervision

### 5. Optimization

**Optimizer**: AdamW
- Learning rate: 1e-4
- Weight decay: 0.01
- Separate LR for different components

**Scheduler**: Cosine annealing with warmup
- Warmup epochs: 5
- Min LR: 1e-6

**Freezing Strategy**:
- VAE: Frozen (pretrained)
- Audio encoder: Frozen (pretrained)
- AR Transformer: Trainable
- Style encoder: Trainable

### 6. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass with teacher forcing
        outputs = model(batch)

        # Compute loss
        loss_dict = criterion(outputs, batch)

        # Backward pass
        loss_dict['loss'].backward()
        optimizer.step()

        # Update learning rate
        scheduler.step()
```

---

## Key Files

### Core Model Files

| File | Description | Lines |
|------|-------------|-------|
| `app/models.py` | BitwiseARModel (main AR model) | 273 |
| `app/transformer.py` | AdaLNSelfAttn, ModifiedSelfAttention | 119 |
| `app/modules/bitwise_vae.py` | VAE encoder/decoder + quantization | 348 |
| `app/modules/style_encoder.py` | Style condition encoder | ~80 |
| `app/modules/wav2vec.py` | Wav2Vec2 audio encoder wrapper | ~60 |

### Training Files

| File | Description | Lines |
|------|-------------|-------|
| `dataset.py` | AudioMotionDataset + collate function | 220 |
| `losses.py` | ARTransformerLoss + combined losses | 200 |
| `train.py` | Main training script | 450 |
| `prepare_dummy_data.py` | Dummy data generation | 120 |

### Documentation

| File | Description |
|------|-------------|
| `TRAINING.md` | Comprehensive training guide |
| `AR_TRANSFORMER_SUMMARY.md` | Architecture summary (this file) |
| `README.md` | Project README |

---

## Quick Start

### 1. Prepare Data

```bash
# Option A: Use your own data
# Organize audio-motion pairs as described in TRAINING.md

# Option B: Generate dummy data for testing
python prepare_dummy_data.py \
    --output_dir dummy_data/train \
    --num_samples 50 \
    --min_duration 2.0 \
    --max_duration 8.0
```

### 2. Train Model

```bash
python train.py \
    --data_root dummy_data/train \
    --config assets/config.json \
    --audio_encoder wav2vec \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --output_dir outputs/ar_training
```

### 3. Monitor Training

```bash
tensorboard --logdir outputs/ar_training/logs
```

### 4. Inference with Trained Model

```python
# Load trained model
checkpoint = torch.load('outputs/ar_training/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Run inference (same as before)
from inference import ARTAvatarInferEngine
engine = ARTAvatarInferEngine(model_path='outputs/ar_training/best_model.pth')
motion = engine.inference(audio_tensor)
```

---

## Implementation Notes

### 1. Key Design Decisions

**Teacher Forcing**:
- Training uses ground truth codes at each level
- Inference uses predicted codes (autoregressive)
- This discrepancy can cause exposure bias
- Mitigation: Label smoothing, sufficient training

**Frozen Components**:
- VAE must be frozen (uses pretrained quantization)
- Audio encoder frozen (reduces overfitting, faster training)
- Only AR Transformer and Style Encoder are trained

**Hierarchical Generation**:
- Coarse-to-fine generation improves quality
- Allows efficient long-sequence modeling
- Each level refines previous predictions

**Binary Quantization**:
- 32 binary dimensions = 2^32 codes
- Compact representation
- Straight-through estimator for gradients

### 2. Training Considerations

**Batch Size**:
- Default: 8 (fits on 16GB GPU)
- Reduce if OOM
- Larger batch → more stable gradients

**Learning Rate**:
- 1e-4 works well for most cases
- Reduce if training unstable
- Increase if converging slowly

**Data Augmentation**:
- Currently: Random volume scaling (0.8-1.2x)
- Can add: Time stretching, pitch shifting, noise injection
- Be careful not to break audio-motion sync

**Sequence Length**:
- Fixed to 100 frames (4 seconds) for simplicity
- Can extend to variable length with padding masks
- Longer sequences require more memory

### 3. Potential Improvements

**Architecture**:
- Add cross-attention between audio and motion
- Use rotary position embeddings (RoPE)
- Try different attention mechanisms (FlashAttention)

**Training**:
- Scheduled sampling (gradually use predictions)
- Adversarial training (discriminator for realism)
- Contrastive learning (audio-motion alignment)

**Data**:
- Multi-speaker training (diversity)
- Cross-lingual training (generalization)
- Data augmentation (robustness)

### 4. Known Limitations

1. **Fixed Chunk Size**: Currently limited to 100-frame chunks
2. **No Long-Range Context**: Context limited to 181 tokens
3. **Exposure Bias**: Teacher forcing vs autoregressive gap
4. **Style Conditioning**: Requires reference motion for style

### 5. Debugging Tips

**Check Data Loading**:
```python
from dataset import AudioMotionDataset
dataset = AudioMotionDataset('dummy_data/train')
sample = dataset[0]
print(sample['audio'].shape, sample['motion'].shape)
```

**Check Forward Pass**:
```python
model = BitwiseARModel(model_cfg)
outputs = model(batch)
print(outputs['logits'].shape)  # [B, 181, 32, 2]
print(outputs['targets'].shape) # [B, 181, 32]
```

**Check Loss**:
```python
criterion = CombinedLoss(patch_nums=[1,5,25,50,100])
loss_dict = criterion(outputs, batch)
print(loss_dict['loss'])
print(loss_dict['accuracy'])
```

---

## Summary

The AR Transformer in ARTalk is a sophisticated architecture for speech-driven facial animation:

- **Hierarchical autoregressive generation** across 5 scales
- **Binary quantization** via pretrained VAE (2^32 codes)
- **Audio conditioning** via frozen Wav2Vec2/MIMI encoder
- **Style control** via learned style embeddings
- **12-layer Transformer** with adaptive LayerNorm

The training implementation provides:

- **Complete training pipeline** from data loading to checkpointing
- **Modular loss functions** with per-level monitoring
- **Flexible optimization** with warmup and scheduling
- **TensorBoard logging** for easy monitoring
- **Comprehensive documentation** for easy adoption

For detailed usage instructions, see `TRAINING.md`.

For questions or issues, please open a GitHub issue.
