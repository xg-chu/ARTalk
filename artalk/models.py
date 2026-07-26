#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .transformer import AdaLNSelfAttn
from .modules import BITWISE_VAE, MimiModelWrapper, Wav2Vec2Model, Wav2Vec2Config, StyleEncoder

class BitwiseARModel(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        # build basic vae
        self.basic_vae = BITWISE_VAE(model_cfg=model_cfg["VAE_CONFIG"])
        self.patch_nums = self.basic_vae.patch_nums
        self.vqfeat_embed = nn.Linear(self.basic_vae.code_dim, 768)
        # style encoder
        self.style_encoder = StyleEncoder()
        self.style_cond_embed = nn.Linear(128, 768)
        # audio encoder
        if model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'wav2vec':
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
            self.audio_encoder = Wav2Vec2Model(config)
            self.audio_feature_dim = 1024
        elif model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'mimi':
            self.audio_encoder = MimiModelWrapper()
            self.audio_feature_dim = 512
        else:
            raise ValueError("Invalid audio encoder: {}".format(model_cfg["AR_CONFIG"]['AUDIO_ENCODER']))
        # autoregressive generator
        self.attn_depth = model_cfg["AR_CONFIG"]['T_DEPTH']
        dpr = [x.item() for x in torch.linspace(0, 0.1 * self.attn_depth/24, self.attn_depth)]
        self.attn_blocks = nn.ModuleList([
            AdaLNSelfAttn(embed_dim=768, cond_dim=self.audio_feature_dim, num_heads=model_cfg["AR_CONFIG"]['T_NUM_HEADS'], drop_path=dpr[depth_idx])
            for depth_idx in range(self.attn_depth)
        ])
        # logits head part
        self.cond_logits_head = AdaLNBeforeHead(embed_dim=768, cond_dim=self.audio_feature_dim)
        self.logits_head = nn.Linear(768, self.basic_vae.code_dim * 2)
        self.null_style_cond = nn.Parameter(torch.randn(1, 1, 768) * 0.5)
        # absolute position and level embedding
        self.prev_ratio = model_cfg["AR_CONFIG"]['PREV_RATIO']
        pos_embed = torch.empty(1, sum(self.patch_nums), 768) # 1, L, C
        nn.init.trunc_normal_(pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.pos_embed = nn.Parameter(pos_embed)
        prev_pos_embed = torch.empty(1, sum(self.patch_nums) * self.prev_ratio, 768)
        nn.init.trunc_normal_(prev_pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.prev_pos_embed = nn.Parameter(prev_pos_embed)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), 768)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / 768 / 3))
        attn_bias_for_masking, lvl_idx = self.build_attn_mask(self.patch_nums)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking)
        self.register_buffer('lvl_idx', lvl_idx)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def inference(self, batch, with_gtmotion=False):
        batch_size = batch["audio"].shape[0]
        assert batch_size == 1, "Only support batch size 1 for inference."
        seq_length = math.ceil(batch["audio"].shape[-1] / 16000 * 25.0)
        if 'style_motion' in batch.keys() and batch['style_motion'] is not None:
            motion_style = self.style_encoder(batch["style_motion"]).detach()
            motion_style_cond = self.style_cond_embed(motion_style)[:, None]
            motion_style_cond = motion_style_cond * 1.1 - self.null_style_cond * 0.1
        else:
            print("No style motion provided, use default style condition.")
            motion_style_cond = self.null_style_cond
        lvl_pos_embed = self.lvl_embed(self.lvl_idx) + self.pos_embed
        prev_lvl_pos_embed = self.lvl_embed(self.lvl_idx).repeat(1, self.prev_ratio, 1) + self.prev_pos_embed

        # padding frames and audios
        padded_frame_length = math.ceil(seq_length / self.patch_nums[-1]) * self.patch_nums[-1]
        padded_audio_length = int(padded_frame_length / 25.0 * 16000)
        patch_audio_length = int(self.patch_nums[-1] / 25.0 * 16000)
        audio_chunks = batch["audio"]
        audio_chunks = torch.cat([
                audio_chunks, audio_chunks.new_zeros(batch_size, padded_audio_length - audio_chunks.shape[1])
            ], dim=-1
        ).split(patch_audio_length, dim=-1)
        prev_motion = batch["audio"].new_zeros(batch_size, self.patch_nums[-1], self.basic_vae.motion_dim)
        prev_code_bits, _ = self.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)
        prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits)
        prev_attn_feat = torch.cat([motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1).repeat(1, self.prev_ratio, 1)
        # split patchs
        all_pred_motions = []
        for idx in range(len(audio_chunks)):
            split_audio_feat = self.audio_encoder(audio_chunks[idx]).permute(0, 2, 1) # B, L, C -> B, C, L
            split_audio_feats = [F.interpolate(split_audio_feat, size=(pn), mode='area').permute(0, 2, 1) for pn in self.patch_nums] # B, L, C
            split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()
            next_ar_vqfeat = motion_style_cond
            for pidx, pn in enumerate(self.patch_nums):
                patch_audio_cond = split_audio_cond[:, :sum(self.patch_nums[:pidx+1])]
                patch_attn_bias = self.attn_bias_for_masking[:, :, :sum(self.patch_nums[:pidx+1]), :sum(self.patch_nums[:pidx+1])+sum(self.patch_nums)*self.prev_ratio]
                attn_feat = next_ar_vqfeat + lvl_pos_embed[:, :next_ar_vqfeat.shape[1]]
                for bidx in range(self.attn_depth):
                    attn_feat = self.attn_blocks[bidx](attn_feat, prev_attn_feat + prev_lvl_pos_embed, patch_audio_cond, attn_bias=patch_attn_bias)
                pred_motion_logits = self.logits_head(self.cond_logits_head(attn_feat, patch_audio_cond))
                pred_motion_bits = pred_motion_logits.view(pred_motion_logits.shape[0], pred_motion_logits.shape[1], -1, 2).argmax(dim=-1)
                if pidx < len(self.patch_nums) - 1:
                    next_ar_vqfeat = self.basic_vae.vqidx_to_ar_vqfeat(pidx, pred_motion_bits)
                    next_ar_vqfeat = torch.cat([motion_style_cond, self.vqfeat_embed(next_ar_vqfeat)], dim=1)
            split_prev_motion, split_pred_motion = self.basic_vae.vqidx_to_motion(prev_code_bits, pred_motion_bits)
            all_pred_motions.append(split_pred_motion)
            # set next
            prev_code_bits, _ = self.basic_vae.quant_to_vqidx(split_pred_motion, this_motion=None)
            prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits).detach()
            this_prev_attn_feat = torch.cat([motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1)
            prev_attn_feat = torch.cat([prev_attn_feat[:, this_prev_attn_feat.shape[1]:], this_prev_attn_feat], dim=1)
        pred_motions = torch.cat(all_pred_motions, dim=1)[:, :seq_length]
        if with_gtmotion:
            min_length = min(batch["motion"].shape[1], pred_motions.shape[1])
            shape_code = batch["shape"].expand(-1, min_length, -1)
            return pred_motions[:, :min_length], batch["motion"][:, :min_length], shape_code
        else:
            return pred_motions

    @torch.no_grad()
    def build_attn_mask(self, patch_nums):
        L = sum(patch_nums)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(patch_nums)]).view(1, L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_idx = dT[:, 0].contiguous()
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous()
        zero_shape = list(attn_bias_for_masking.shape)
        zero_shape[-1] = patch_nums[-1]
        zero_attn_bias_for_masking = attn_bias_for_masking.new_zeros(attn_bias_for_masking.shape)
        zero_attn_bias_for_masking = zero_attn_bias_for_masking.repeat(1, 1, 1, self.prev_ratio)
        attn_bias_for_masking = torch.cat([zero_attn_bias_for_masking, attn_bias_for_masking], dim=-1)
        return attn_bias_for_masking, lvl_idx


class AdaLNBeforeHead(nn.Module):
    def __init__(self, embed_dim, cond_dim):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = embed_dim, cond_dim
        self.ln_wo_grad = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(cond_dim, 2*embed_dim))
    
    def forward(self, feat, cond_BD):
        batch_size, cond_len = feat.shape[0], cond_BD.shape[1]
        scale, shift = self.ada_lin(cond_BD).view(batch_size, cond_len, 2, -1).unbind(2)
        return self.ln_wo_grad(feat).mul(scale.add(1)).add_(shift)


def sample_with_top_k_top_p_(logits_BLV, top_k=2, top_p=0.95, num_samples=1):  # return idx, shaped (B, L)
    B, L, V = logits_BLV.shape
    if top_k > 0:
        idx_to_remove = logits_BLV < logits_BLV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BLV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BLV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BLV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BLV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=None).view(B, L, num_samples)[:, :, 0]

