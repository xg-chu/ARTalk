#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Loss functions for AR Transformer Training

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARTransformerLoss(nn.Module):
    """Loss function for autoregressive transformer training.

    Combines:
    1. Cross-entropy loss for binary code prediction
    2. Optional label smoothing
    3. Optional per-level weighting
    """

    def __init__(
        self,
        patch_nums=[1, 5, 25, 50, 100],
        label_smoothing=0.0,
        level_weights=None,
        reduce='mean'
    ):
        super().__init__()
        self.patch_nums = patch_nums
        self.label_smoothing = label_smoothing
        self.reduce = reduce

        # Level weights for hierarchical loss
        if level_weights is None:
            # Default: equal weight for all levels
            level_weights = [1.0] * len(patch_nums)
        assert len(level_weights) == len(patch_nums), "level_weights must match patch_nums"
        self.register_buffer('level_weights', torch.tensor(level_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        """Compute cross-entropy loss for binary code prediction.

        Args:
            logits: [B, sum(patch_nums), code_dim, 2] predicted logits
            targets: [B, sum(patch_nums), code_dim] target binary codes (0 or 1)

        Returns:
            dict with loss components
        """
        B, L, C, _ = logits.shape

        # Reshape for cross-entropy loss
        logits_flat = logits.reshape(-1, 2)  # [B*L*C, 2]
        targets_flat = targets.reshape(-1).long()  # [B*L*C]

        # Compute cross-entropy loss
        if self.label_smoothing > 0:
            # Label smoothing
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction='none',
                label_smoothing=self.label_smoothing
            )
        else:
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction='none'
            )

        # Reshape loss back to [B, L, C]
        loss = loss.reshape(B, L, C)

        # Compute per-level loss for monitoring
        level_losses = []
        start_idx = 0
        for pidx, pn in enumerate(self.patch_nums):
            end_idx = start_idx + pn
            level_loss = loss[:, start_idx:end_idx].mean()
            level_losses.append(level_loss)
            start_idx = end_idx

        # Weighted loss across levels
        weighted_loss = sum(w * l for w, l in zip(self.level_weights, level_losses))

        # Total loss
        if self.reduce == 'mean':
            total_loss = weighted_loss
        elif self.reduce == 'sum':
            total_loss = weighted_loss * sum(self.patch_nums) * C
        else:
            raise ValueError(f"Invalid reduce mode: {self.reduce}")

        # Compute accuracy
        with torch.no_grad():
            pred_codes = logits.argmax(dim=-1)
            accuracy = (pred_codes == targets).float().mean()

            # Per-level accuracy
            level_accs = []
            start_idx = 0
            for pidx, pn in enumerate(self.patch_nums):
                end_idx = start_idx + pn
                level_acc = (pred_codes[:, start_idx:end_idx] == targets[:, start_idx:end_idx]).float().mean()
                level_accs.append(level_acc)
                start_idx = end_idx

        return {
            'loss': total_loss,
            'accuracy': accuracy,
            'level_losses': torch.stack(level_losses),
            'level_accs': torch.stack(level_accs),
        }


class MotionReconstructionLoss(nn.Module):
    """Optional auxiliary loss for motion reconstruction quality."""

    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_motion, gt_motion):
        """Compute reconstruction loss.

        Args:
            pred_motion: [B, T, 106] predicted motion
            gt_motion: [B, T, 106] ground truth motion

        Returns:
            loss: scalar
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_motion, gt_motion)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_motion, gt_motion)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_motion, gt_motion)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        return loss


class CombinedLoss(nn.Module):
    """Combined loss for AR Transformer training.

    Combines:
    1. AR cross-entropy loss (main)
    2. Motion reconstruction loss (auxiliary, optional)
    """

    def __init__(
        self,
        patch_nums=[1, 5, 25, 50, 100],
        label_smoothing=0.0,
        level_weights=None,
        recon_loss_weight=0.0,
        recon_loss_type='l1'
    ):
        super().__init__()
        self.ar_loss = ARTransformerLoss(
            patch_nums=patch_nums,
            label_smoothing=label_smoothing,
            level_weights=level_weights
        )
        self.recon_loss_weight = recon_loss_weight
        if recon_loss_weight > 0:
            self.recon_loss = MotionReconstructionLoss(loss_type=recon_loss_type)

    def forward(self, outputs, batch):
        """Compute combined loss.

        Args:
            outputs: dict from model forward pass
            batch: dict with ground truth data

        Returns:
            dict with loss components
        """
        # Main AR loss
        ar_loss_dict = self.ar_loss(outputs['logits'], outputs['targets'])

        total_loss = ar_loss_dict['loss']
        loss_dict = {
            'ar_loss': ar_loss_dict['loss'],
            'accuracy': ar_loss_dict['accuracy'],
            'level_losses': ar_loss_dict['level_losses'],
            'level_accs': ar_loss_dict['level_accs'],
        }

        # Optional reconstruction loss
        if self.recon_loss_weight > 0:
            recon_loss = self.recon_loss(
                outputs['motion_recon'],
                batch['motion'][:, :outputs['motion_recon'].shape[1]]
            )
            total_loss = total_loss + self.recon_loss_weight * recon_loss
            loss_dict['recon_loss'] = recon_loss

        loss_dict['loss'] = total_loss

        return loss_dict


# Testing
if __name__ == '__main__':
    # Test AR loss
    patch_nums = [1, 5, 25, 50, 100]
    B, C = 4, 32
    L = sum(patch_nums)

    logits = torch.randn(B, L, C, 2)
    targets = torch.randint(0, 2, (B, L, C))

    ar_loss = ARTransformerLoss(patch_nums=patch_nums, label_smoothing=0.1)
    loss_dict = ar_loss(logits, targets)

    print("AR Loss Test:")
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Accuracy: {loss_dict['accuracy'].item():.4f}")
    print(f"  Level losses: {loss_dict['level_losses']}")
    print(f"  Level accs: {loss_dict['level_accs']}")

    # Test combined loss
    pred_motion = torch.randn(B, 100, 106)
    gt_motion = torch.randn(B, 100, 106)

    outputs = {
        'logits': logits,
        'targets': targets,
        'motion_recon': pred_motion,
    }
    batch = {
        'motion': gt_motion,
    }

    combined_loss = CombinedLoss(
        patch_nums=patch_nums,
        label_smoothing=0.1,
        recon_loss_weight=0.1
    )
    loss_dict = combined_loss(outputs, batch)

    print("\nCombined Loss Test:")
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  AR loss: {loss_dict['ar_loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  Accuracy: {loss_dict['accuracy'].item():.4f}")

    print("\nLoss functions test passed!")
