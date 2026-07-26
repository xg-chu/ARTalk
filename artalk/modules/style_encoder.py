#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import torch.nn as nn
from .data_stats import ALLTALKEMICA_MEAN, ALLTALKEMICA_STD

class StyleEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # amespace(mode='train', iter=100000, rot_repr='aa', no_head_pose=False, feature_dim=128, n_heads=4, n_layers=4, mlp_ratio=4, n_motions=100, fps=25, data_root=PosixPath('datasets/HDTF_THHQ/lmdb'), stats_file=PosixPath('stats_train.npz'), batch_size=32, num_workers=4, exp_name='head-L4H4_T0.1_BS32', max_iter=100000, lr=0.0001, temperature=0.1, save_iter=2000, val_iter=2000, log_iter=50, log_smooth_win=50)
        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, activation='gelu', batch_first=True
        )
        self.PE = PositionalEncoding(128)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(106, 128),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=4),
        })
        self.register_buffer("motion_mean", torch.tensor(ALLTALKEMICA_MEAN).float())
        self.register_buffer("motion_std", torch.tensor(ALLTALKEMICA_STD).float())

    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        motion_coef = self.norm_with_stats(motion_coef)
        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)
        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)
        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)
        return feat

    def norm_with_stats(self, motion_coef):
        normed_motion_coef = (motion_coef.clone() - self.motion_mean) / self.motion_std
        return normed_motion_coef


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)
