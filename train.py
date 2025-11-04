#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Training script for AR Transformer

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from app import BitwiseARModel
from dataset import AudioMotionDataset, AudioMotionCollateFn
from losses import CombinedLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train AR Transformer')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of training data')
    parser.add_argument('--metadata_file', type=str, default='metadata.json',
                        help='Metadata file name')
    parser.add_argument('--val_data_root', type=str, default=None,
                        help='Root directory of validation data (optional)')

    # Model
    parser.add_argument('--config', type=str, default='assets/config.json',
                        help='Model config file')
    parser.add_argument('--audio_encoder', type=str, default='wav2vec',
                        choices=['wav2vec', 'mimi'],
                        help='Audio encoder type')
    parser.add_argument('--pretrained_vae', type=str, default=None,
                        help='Path to pretrained VAE checkpoint')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained AR model checkpoint (for resume)')
    parser.add_argument('--freeze_vae', action='store_true', default=True,
                        help='Freeze VAE during training')
    parser.add_argument('--freeze_audio_encoder', action='store_true', default=True,
                        help='Freeze audio encoder during training')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'step', 'constant'],
                        help='Learning rate schedule')

    # Loss
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for cross-entropy loss')
    parser.add_argument('--recon_loss_weight', type=float, default=0.0,
                        help='Weight for motion reconstruction loss')

    # Logging
    parser.add_argument('--output_dir', type=str, default='outputs/ar_training',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval in steps')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval in epochs')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval in epochs')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_optimizer(model, args, freeze_vae=True, freeze_audio_encoder=True):
    """Create optimizer with different learning rates for different components."""
    param_groups = []

    # Separate parameters by component
    vae_params = []
    audio_params = []
    transformer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'basic_vae' in name:
            if freeze_vae:
                param.requires_grad = False
            else:
                vae_params.append(param)
        elif 'audio_encoder' in name:
            if freeze_audio_encoder:
                param.requires_grad = False
            else:
                audio_params.append(param)
        elif 'attn_blocks' in name or 'logits_head' in name or 'cond_logits_head' in name:
            transformer_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups with different learning rates
    if len(transformer_params) > 0:
        param_groups.append({'params': transformer_params, 'lr': args.lr})
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': args.lr})
    if len(vae_params) > 0:
        param_groups.append({'params': vae_params, 'lr': args.lr * 0.1})
    if len(audio_params) > 0:
        param_groups.append({'params': audio_params, 'lr': args.lr * 0.1})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def create_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler."""
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01
        )
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=steps_per_epoch * 10, gamma=0.5
        )
    else:  # constant
        scheduler = None

    # Wrap with warmup
    if warmup_steps > 0:
        def warmup_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        return warmup_scheduler, scheduler
    else:
        return None, scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler_warmup, scheduler_main, scaler, epoch, args, writer, global_step):
    """Train for one epoch."""
    model.train()

    # Freeze components if specified
    if args.freeze_vae:
        model.basic_vae.eval()
        for param in model.basic_vae.parameters():
            param.requires_grad = False

    if args.freeze_audio_encoder:
        model.audio_encoder.eval()
        for param in model.audio_encoder.parameters():
            param.requires_grad = False

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    running_acc = 0.0

    for step, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
        else:
            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update learning rate
        if scheduler_warmup is not None and global_step < args.warmup_epochs * len(dataloader):
            scheduler_warmup.step()
        elif scheduler_main is not None:
            scheduler_main.step()

        # Update metrics
        running_loss += loss.item()
        running_acc += loss_dict['accuracy'].item()

        # Log
        if step % args.log_interval == 0 and step > 0:
            avg_loss = running_loss / args.log_interval
            avg_acc = running_acc / args.log_interval
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}',
                'lr': f'{lr:.2e}'
            })

            # Tensorboard logging
            writer.add_scalar('train/loss', avg_loss, global_step)
            writer.add_scalar('train/accuracy', avg_acc, global_step)
            writer.add_scalar('train/ar_loss', loss_dict['ar_loss'].item(), global_step)
            writer.add_scalar('train/lr', lr, global_step)

            if 'recon_loss' in loss_dict:
                writer.add_scalar('train/recon_loss', loss_dict['recon_loss'].item(), global_step)

            # Log per-level losses and accuracies
            for i, (l_loss, l_acc) in enumerate(zip(loss_dict['level_losses'], loss_dict['level_accs'])):
                writer.add_scalar(f'train/level_{i}_loss', l_loss.item(), global_step)
                writer.add_scalar(f'train/level_{i}_acc', l_acc.item(), global_step)

            running_loss = 0.0
            running_acc = 0.0

        global_step += 1

    return global_step


@torch.no_grad()
def validate(model, dataloader, criterion, epoch, args, writer):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Validation {epoch}')
    for batch in pbar:
        # Move to device
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        outputs = model(batch)
        loss_dict = criterion(outputs, batch)

        # Update metrics
        total_loss += loss_dict['loss'].item()
        total_acc += loss_dict['accuracy'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss_dict["loss"].item():.4f}',
            'acc': f'{loss_dict["accuracy"].item():.4f}'
        })

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    # Log to tensorboard
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/accuracy', avg_acc, epoch)

    print(f'Validation Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')

    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, global_step, args, best=False):
    """Save checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }

    if best:
        path = os.path.join(args.output_dir, 'best_model.pth')
    else:
        path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')

    torch.save(checkpoint, path)
    print(f'Saved checkpoint to {path}')


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load config
    with open(args.config, 'r') as f:
        model_cfg = json.load(f)

    # Add audio encoder to config
    model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] = args.audio_encoder

    # Create datasets
    train_dataset = AudioMotionDataset(
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        chunk_duration=4.0,
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=AudioMotionCollateFn(),
        pin_memory=True
    )

    # Validation dataset (optional)
    val_loader = None
    if args.val_data_root is not None:
        val_dataset = AudioMotionDataset(
            data_root=args.val_data_root,
            metadata_file=args.metadata_file,
            chunk_duration=4.0,
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=AudioMotionCollateFn(),
            pin_memory=True
        )

    # Create model
    model = BitwiseARModel(model_cfg=model_cfg)

    # Load pretrained VAE if provided
    if args.pretrained_vae is not None:
        print(f'Loading pretrained VAE from {args.pretrained_vae}')
        vae_checkpoint = torch.load(args.pretrained_vae, map_location='cpu')
        model.basic_vae.load_state_dict(vae_checkpoint['model_state_dict'], strict=False)

    # Load pretrained model if provided (for resume)
    start_epoch = 0
    global_step = 0
    if args.pretrained_model is not None:
        print(f'Loading pretrained model from {args.pretrained_model}')
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)

    model = model.to(args.device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args, args.freeze_vae, args.freeze_audio_encoder)
    scheduler_warmup, scheduler_main = create_scheduler(optimizer, args, len(train_loader))

    # Load optimizer state if resuming
    if args.pretrained_model is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Create loss
    criterion = CombinedLoss(
        patch_nums=model_cfg["VAE_CONFIG"]["V_PATCH_NUMS"],
        label_smoothing=args.label_smoothing,
        recon_loss_weight=args.recon_loss_weight
    )

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler_warmup, scheduler_main, scaler,
            epoch, args, writer, global_step
        )

        # Validate
        if val_loader is not None and (epoch + 1) % args.eval_interval == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, epoch, args, writer)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, global_step, args, best=True)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, global_step, args, best=False)

    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()
