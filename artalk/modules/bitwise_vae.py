#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# from .flame_model import FLAMEModel, RenderMesh
from .data_stats import TFHP_MEAN, TFHP_STD, ALLTALKEMICA_MEAN, ALLTALKEMICA_STD

class BITWISE_VAE(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        self.motion_dim = 106
        self.code_dim = model_cfg['V_CODE_DIM']
        self.patch_nums = model_cfg['V_PATCH_NUMS']
        self.encoder = TransformerEncoder(
            inp_dim=self.motion_dim, hidden_dim=model_cfg['T_HIDDEN_DIM'], code_dim=self.code_dim, depth=model_cfg['T_DEPTH'], n_heads=model_cfg['T_NUM_HEADS']
        )
        self.decoder = TransformerDecoder(
            code_dim=self.code_dim, hidden_dim=model_cfg['T_HIDDEN_DIM'], out_dim=self.motion_dim, depth=model_cfg['T_DEPTH'], n_heads=model_cfg['T_NUM_HEADS']
        )
        self.quantize = MultiScaleBSQ(codebook_dim=self.code_dim, scale_schedule=self.patch_nums)
        attn_mask = self.build_attn_mask(self.patch_nums[-1])
        self.register_buffer('attn_mask', attn_mask)

        # absolute position and level embedding
        enc_pos_embed = torch.empty(1, self.patch_nums[-1]*2, self.motion_dim) # 1, L, C
        nn.init.trunc_normal_(enc_pos_embed, mean=0, std=math.sqrt(1 / self.motion_dim / 3))
        self.enc_pos_embed = nn.Parameter(enc_pos_embed)
        dec_pos_embed = torch.empty(1, self.patch_nums[-1]*2, self.code_dim) # 1, L, C
        nn.init.trunc_normal_(dec_pos_embed, mean=0, std=math.sqrt(1 / self.code_dim / 3))
        self.dec_pos_embed = nn.Parameter(dec_pos_embed)

        # stat & render
        self.register_buffer("motion_mean", torch.tensor(ALLTALKEMICA_MEAN).float())
        self.register_buffer("motion_std", torch.tensor(ALLTALKEMICA_STD).float())

    def get_flame_verts(self, flame_model, shape_params, motion_params, with_global=False):
        exp_code, pose_code = motion_params[..., :100], motion_params[..., 100:]
        if not with_global:
            pose_code = torch.cat([torch.zeros_like(pose_code[..., :3]), pose_code[..., 3:]], dim=-1)
        if shape_params.dim() == 2:
            verts = flame_model(shape_params=shape_params, expression_params=exp_code, pose_params=pose_code)
        elif shape_params.dim() == 3:
            verts = []
            for bidx in range(shape_params.shape[0]):
                this_verts = flame_model(shape_params=shape_params[bidx], expression_params=exp_code[bidx], pose_params=pose_code[bidx])
                verts.append(this_verts)
            verts = torch.stack(verts, dim=0)
        else:
            raise ValueError("Invalid shape of shape_params: {}".format(shape_params.shape))
        return verts

    def norm_with_stats(self, motion_code):
        normed_motion_code = (motion_code - self.motion_mean) / self.motion_std
        return normed_motion_code

    def unnorm_with_stats(self, motion_code):
        unnormed_motion_code = motion_code * self.motion_std + self.motion_mean
        return unnormed_motion_code

    @torch.no_grad()
    def build_attn_mask(self, patch_nums):
        zero_attn_bias_block = torch.zeros(patch_nums, patch_nums)
        minf_attn_bias_block = torch.ones(patch_nums, patch_nums) * (-torch.inf)
        attn_mask = torch.cat([
                torch.cat([zero_attn_bias_block, minf_attn_bias_block], dim=-1),
                torch.cat([zero_attn_bias_block, zero_attn_bias_block], dim=-1)
            ], dim=0
        )
        return attn_mask[None, None]

    @torch.no_grad()
    def quant_to_vqidx(self, prev_motion, this_motion=None):
        seq_len = self.patch_nums[-1]
        if this_motion is not None:
            all_motion = torch.cat([prev_motion, this_motion], dim=1)
            enc_in = self.norm_with_stats(all_motion)
            enc_out = self.encoder(enc_in+self.enc_pos_embed, attn_mask=self.attn_mask)
            prev_enc_out, this_enc_out = enc_out[:, :seq_len], enc_out[:, seq_len:]
            _, prev_code_idx, _ = self.quantize(prev_enc_out)
            _, this_code_idx, _ = self.quantize(this_enc_out)
        else:
            enc_in = self.norm_with_stats(prev_motion)
            enc_out = self.encoder(enc_in+self.enc_pos_embed[:, :seq_len], attn_mask=self.attn_mask[:, :, :seq_len, :seq_len])
            _, prev_code_idx, _ = self.quantize(enc_out)
            this_code_idx = None
        return prev_code_idx, this_code_idx

    @torch.no_grad()
    def flip_quant_to_vqidx(self, prev_motion, this_motion, flip_ratio):
        seq_len = self.patch_nums[-1]
        all_motion = torch.cat([prev_motion, this_motion], dim=1)
        enc_in = self.norm_with_stats(all_motion)
        enc_out = self.encoder(enc_in+self.enc_pos_embed, attn_mask=self.attn_mask)
        this_enc_out = enc_out[:, seq_len:]
        _, this_code_idx = self.quantize.flip_quant_to_vqidx(this_enc_out, flip_ratio)
        return this_code_idx

    @torch.no_grad()
    def vqidx_to_motion(self, prev_code_idx, this_code_idx):
        seq_len = self.patch_nums[-1]
        prev_vq_out = self.quantize.vqidx_to_feat(prev_code_idx, multi_scale=False)
        this_vq_out = self.quantize.vqidx_to_feat(this_code_idx, multi_scale=False)
        vq_out = torch.cat([prev_vq_out, this_vq_out], dim=1)
        dec_out = self.decoder(vq_out+self.dec_pos_embed, attn_mask=self.attn_mask)
        motion_code = self.unnorm_with_stats(dec_out)
        return motion_code[:, :seq_len], motion_code[:, seq_len:]

    # for training of var model
    @torch.no_grad()
    def vqidx_to_ms_vqfeat(self, code_idx):
        vqfeat = self.quantize.vqidx_to_feat(code_idx, multi_scale=True)
        return vqfeat

    # for inference of var model
    @torch.no_grad()
    def vqidx_to_ar_vqfeat(self, pidx, code_idx):
        next_ar_vqfeat = self.quantize.vqidx_to_ar_vqfeat(pidx, code_idx)
        return next_ar_vqfeat


class TransformerEncoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, code_dim, depth=6, n_heads=8):
        super().__init__()
        self.inp_mapping = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.code_mapping = nn.Linear(hidden_dim, code_dim)
        # transformer
        blocks = []
        for i in range(depth):
            blocks += [
                SimpleSelfAttention(hidden_dim, n_heads=n_heads),
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, int(1.5 * hidden_dim)),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(int(1.5 * hidden_dim), hidden_dim)
                )
            ]
        self.encoder_transformer = nn.ModuleList(blocks)

    def forward(self, inp_BLC, attn_mask=None):
        feat = self.inp_mapping(inp_BLC)
        for block in self.encoder_transformer:
            if isinstance(block, SimpleSelfAttention):
                feat = feat + block(feat, attn_mask)
            else:
                feat = feat + block(feat)
        out = self.code_mapping(feat)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, code_dim, hidden_dim, out_dim, depth=6, n_heads=8):
        super().__init__()
        self.inp_mapping = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.out_mapping = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.out_mapping.weight, gain=0.05)
        nn.init.constant_(self.out_mapping.bias, 0)
        # transformer
        blocks = []
        for i in range(depth):
            blocks += [
                SimpleSelfAttention(hidden_dim, n_heads=n_heads),
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, int(1.5 * hidden_dim)),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(int(1.5 * hidden_dim), hidden_dim)
                )
            ]
        self.decoder_transformer = nn.ModuleList(blocks)

    def forward(self, inp_BLC, attn_mask=None):
        feat = self.inp_mapping(inp_BLC)
        for block in self.decoder_transformer:
            if isinstance(block, SimpleSelfAttention):
                feat = feat + block(feat, attn_mask)
            else:
                feat = feat + block(feat)
        out = self.out_mapping(feat)
        return out


class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.scale = int(hidden_dim)**(-0.5)
        self.rearrange_qkv = Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=self.n_heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        qkv = self.to_qkv(self.norm(x)) # [B, L, C]
        q, k, v = self.rearrange_qkv(qkv).unbind(0) # [B, L, C] -> [B, H, L, c]
        # compute attention
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=self.scale, attn_mask=attn_mask, dropout_p=0.0
        )
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out


class MultiScaleBSQ(nn.Module):
    def __init__(self, codebook_dim=32, scale_schedule=None):
        super().__init__()
        # codebook size -> 2 ** codebook_dim
        self.codebook_dim = codebook_dim
        self.scale_lvls = len(scale_schedule)
        self.scale_schedule = scale_schedule
        self.bsq_quant = BSQ(codebook_dim=codebook_dim)

    def forward(self, f_BTC):
        B, T, C = f_BTC.size()
        quantized_out, residual = 0., f_BTC
        all_losses, all_bit_indices = [], []
        for lvl_idx, pt in enumerate(self.scale_schedule):
            interpolate_residual = F.interpolate(residual.permute(0, 2, 1), size=(pt), mode='area').permute(0, 2, 1).contiguous() if pt != T else residual
            quantized, bit_indices, loss = self.bsq_quant(interpolate_residual)
            quantized = F.interpolate(quantized.permute(0, 2, 1), size=(T), mode='linear').permute(0, 2, 1).contiguous() if pt != T else quantized
            residual = residual - quantized.detach() # remove_residual_detach = False
            quantized_out = quantized_out + quantized
            all_bit_indices.append(bit_indices)
            all_losses.append(loss)
        # stack all losses and indices
        all_losses = torch.stack(all_losses, dim=-1)
        all_bit_indices = torch.cat(all_bit_indices, dim=1)
        return quantized_out, all_bit_indices, all_losses

    @torch.no_grad()
    def flip_quant_to_vqidx(self, f_BTC, flip_ratio):
        B, T, C = f_BTC.size()
        quantized_out, residual = 0., f_BTC
        all_bit_indices = []
        for lvl_idx, pt in enumerate(self.scale_schedule):
            interpolate_residual = F.interpolate(residual.permute(0, 2, 1), size=(pt), mode='area').permute(0, 2, 1).contiguous() if pt != T else residual
            quantized, bit_indices, _ = self.bsq_quant(interpolate_residual)
            mask_flip = torch.rand(bit_indices.shape).to(bit_indices.device) < flip_ratio
            pred_bit_indices = bit_indices.clone()
            # if lvl_idx < self.scale_lvls-1:
            pred_bit_indices[mask_flip] = 1 - pred_bit_indices[mask_flip]
            quantized = (pred_bit_indices.float() * 2 - 1.0) / (self.codebook_dim ** 0.5)
            quantized = F.interpolate(quantized.permute(0, 2, 1), size=(T), mode='linear').permute(0, 2, 1).contiguous() if pt != T else quantized
            residual = residual - quantized.detach() # remove_residual_detach = False
            quantized_out = quantized_out + quantized
            all_bit_indices.append(pred_bit_indices)
        all_bit_indices = torch.cat(all_bit_indices, dim=1)
        return quantized_out, all_bit_indices

    @torch.no_grad()
    def vqidx_to_feat(self, bit_indices, multi_scale=False):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        ori_h_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim ** 0.5)
        pn_start, pn_next = 0, self.scale_schedule[0]
        if multi_scale:
            ori_h_BCT = ori_h_BTC.permute(0, 2, 1).contiguous()
            f_hat = bit_indices.new_zeros(B, C, T, dtype=torch.float32)
            next_scales = []
            for pidx in range(self.scale_lvls-1):
                h_BCT = F.interpolate(ori_h_BCT[..., pn_start:pn_next], size=(T), mode='linear')
                f_hat.add_(h_BCT)
                pn_start = pn_next
                pn_next = pn_next + self.scale_schedule[pidx+1]
                next_scales.append(F.interpolate(f_hat, size=(self.scale_schedule[pidx+1]), mode='area'))
            return torch.cat(next_scales, dim=-1).permute(0, 2, 1).contiguous()
        else:
            f_hat = bit_indices.new_zeros(B, T, C, dtype=torch.float32)
            for pidx in range(self.scale_lvls-1):
                h_BCT = F.interpolate(ori_h_BTC[:, pn_start:pn_next].permute(0, 2, 1).contiguous(), size=(T), mode='linear')
                f_hat.add_(h_BCT.permute(0, 2, 1).contiguous())
                pn_start = pn_next
                pn_next = pn_next + self.scale_schedule[pidx+1]
            f_hat.add_(ori_h_BTC[:, pn_start:])
            return f_hat

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    @torch.no_grad()
    def vqidx_to_ar_vqfeat(self, this_pidx, bit_indices): # only used in VAR inference
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        f_hat = bit_indices.new_zeros(B, C, T, dtype=torch.float32)
        ori_h_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim ** 0.5)
        ori_h_BCT = ori_h_BTC.permute(0, 2, 1).contiguous()
        pn_start, pn_next = 0, self.scale_schedule[0]
        next_scales = []
        for pidx in range(this_pidx+1):
            h_BCL = F.interpolate(ori_h_BCT[..., pn_start:pn_next], size=(T), mode='linear').contiguous()
            f_hat.add_(h_BCL)
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx+1]
            next_scales.append(F.interpolate(f_hat.clone(), size=(self.scale_schedule[pidx+1]), mode='area').contiguous())
        return torch.cat(next_scales, dim=-1).permute(0, 2, 1).contiguous()


class BSQ(nn.Module):
    def __init__(self, codebook_dim=32):
        super().__init__()
        self.inv_temperature = 100.0
        self.commit_loss_weight = 0.2
        self.entropy_loss_weight = 0.1
        self.codebook_dim = codebook_dim

    def forward(self, f_BTC):
        f_BTC = F.normalize(f_BTC, dim=-1)
        # use straight-through gradients (optionally with custom activation fn) if training
        quantized = self.quantize(f_BTC) # B, T, C 
        # calculate loss
        persample_entropy, cb_entropy = self.soft_entropy_loss(f_BTC)
        entropy_penalty = (persample_entropy - cb_entropy) / self.inv_temperature
        commit_loss = torch.mean(((quantized.detach() - f_BTC) ** 2).sum(dim=-1))
        aux_loss = entropy_penalty * self.entropy_loss_weight + commit_loss * self.commit_loss_weight
        # gather the indices
        bit_indices = (quantized > 0).int() # B, T, C
        return quantized, bit_indices, aux_loss

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dim, f"Expected {self.codebook_dim} dimensions, got {z.shape[-1]}"
        q_scale = 1. / (self.codebook_dim ** 0.5)
        zhat = torch.where(z > 0, torch.tensor(1).type_as(z), torch.tensor(-1).type_as(z))
        zhat = q_scale * zhat # on unit sphere
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        def get_entropy(count, dim=-1):
            H = -(count * torch.log(count + 1e-8)).sum(dim=dim)
            return H

        p = torch.sigmoid(-4 * z / (self.codebook_dim ** 0.5) * self.inv_temperature)
        prob = torch.stack([p, 1-p], dim=-1) # (b, l, codebook_dim, 2)
        per_sample_entropy = get_entropy(prob, dim=-1).sum(dim=-1).mean() # (b,l, codebook_dim)->(b,l)->scalar
        # macro average of the probability of each subgroup
        avg_prob = prob.mean(dim=[0, 1]) # (codebook_dim, 2)
        codebook_entropy = get_entropy(avg_prob, dim=-1)
        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum()
