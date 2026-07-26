#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from https://github.com/FoundationVision/VAR.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNSelfAttn(nn.Module):
    def __init__(self, embed_dim, cond_dim, num_heads, mlp_ratio=4., drop_path=0., attn_l2_norm=True):
        super(AdaLNSelfAttn, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        hidden_features = round(embed_dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = ModifiedSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_l2_norm=attn_l2_norm)
        self.ffn = torch.nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_features, embed_dim)
        )
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(cond_dim, 6*embed_dim)
        )
        self.ln_wo_grad = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
    def forward(self, feat, prev_feat, cond_BD, attn_bias=None):   # C: embed_dim, D: cond_dim
        batch_size, cond_len = feat.shape[0], cond_BD.shape[1]
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(batch_size, cond_len, 6, -1).unbind(2)
        feat = feat + self.drop_path(
            self.attn(
                self.ln_wo_grad(feat).mul(scale1.add(1)).add_(shift1), prev_feat, attn_bias
            ).mul_(gamma1)
        )
        feat = feat + self.drop_path(
            self.ffn(
                self.ln_wo_grad(feat).mul(scale2.add(1)).add_(shift2)
            ).mul(gamma2)
        )
        return feat


class ModifiedSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_l2_norm=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        if attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 4.0).log(), requires_grad=True)
            self.max_scale_mul = math.log(100)
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
    
    def forward(self, feat, prev_feat, attn_bias):
        B, L, C = feat.shape
        _, prev_L, C = prev_feat.shape
        q = self.query(feat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # BLC => BHLC => BHLc
        k = self.key(torch.cat([prev_feat, feat], dim=1)).view(B, prev_L+L, self.num_heads, self.head_dim).transpose(1, 2)  # BLC => BHLC => BHLc
        v = self.value(torch.cat([prev_feat, feat], dim=1)).view(B, prev_L+L, self.num_heads, self.head_dim).transpose(1, 2)  # BLC => BHLC => BHLc
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=0.0
        ).transpose(1, 2).reshape(B, L, C)
        output = self.proj(output)
        return output


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: 
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


if __name__ == '__main__':
    patch_nums=(1, 5, 25, 50, 100)
    L = sum(patch_nums)
    d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(patch_nums)]).view(1, L, 1)
    dT = d.transpose(1, 2)    # dT: 11L
    # lvl_1L = dT[:, 0].contiguous()
    # self.register_buffer('lvl_1L', lvl_1L)
    attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous().cuda()
    # self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
    # import ipdb; ipdb.set_trace()

    embed_dim = 1024
    depth, num_heads = 16, 16
    dpr = [x.item() for x in torch.linspace(0, 0.1 * depth/24, depth)]

    block_idx = 1
    model = AdaLNSelfAttn(embed_dim=embed_dim, cond_dim=embed_dim, num_heads=num_heads, drop_path=dpr[block_idx]).cuda()
    x = torch.rand(4, 181, 1024).cuda()
    cond = torch.rand(4, 181, 1024).cuda()
    res = model(x, cond, attn_bias_for_masking)
    import ipdb; ipdb.set_trace()
