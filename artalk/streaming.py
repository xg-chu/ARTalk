#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Streaming pieces for ARTalk inference.

This module hosts two streaming-friendly counterparts to the one-shot
inference pipeline; see ``docs/realtime.md`` for the design rationale.

* :class:`ARTalkStreamer` — streaming wrapper around
  ``BitwiseARModel.inference``. The released model processes audio in
  fixed 4-second / 100-frame chunks, and chunk-to-chunk state (previous
  code bits and the rolling attention-feature buffer) is the only thing
  that crosses chunk boundaries; this class exposes that chunked
  computation as a stateful ``feed`` / ``finish`` API. Output is
  bit-exact with one-shot ``BitwiseARModel.inference``.

* :class:`CausalSavgolSmoother` — streaming counterpart to
  ``ARTAvatarInferEngine.smooth_motion_savgol``. Adds a 4-frame
  (160 ms at 25 fps) emission delay in exchange for output that is
  bit-exact with the one-shot smoother.

``scripts/check_streaming_parity.py`` asserts both parity properties.

Other engine post-processing (eye-channel zeroing, ``fix_pose``,
``clip_length`` truncation) is not streaming-aware and stays at the
engine layer.
"""

import math

import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter


SAMPLE_RATE = 16000
FPS = 25.0


class ARTalkStreamer:
    def __init__(self, model, style_motion=None):
        self.model = model
        self.patch_audio_length = int(model.patch_nums[-1] / FPS * SAMPLE_RATE)
        self.frames_per_chunk = model.patch_nums[-1]
        self.motion_dim = model.basic_vae.motion_dim
        self.set_style(style_motion)
        self.reset()

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def set_style(self, style_motion=None):
        model = self.model
        if style_motion is not None:
            if style_motion.dim() == 2:
                style_motion = style_motion[None]
            style_motion = style_motion.to(device=self.device, dtype=self.dtype)
            motion_style = model.style_encoder(style_motion).detach()
            motion_style_cond = model.style_cond_embed(motion_style)[:, None]
            motion_style_cond = motion_style_cond * 1.1 - model.null_style_cond * 0.1
        else:
            motion_style_cond = model.null_style_cond
        self._motion_style_cond = motion_style_cond

    @torch.no_grad()
    def reset(self):
        model = self.model
        prev_motion = torch.zeros(
            1, self.frames_per_chunk, self.motion_dim,
            dtype=self.dtype, device=self.device,
        )
        prev_code_bits, _ = model.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)
        prev_vqfeat = model.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits)
        prev_attn_feat = torch.cat(
            [self._motion_style_cond, model.vqfeat_embed(prev_vqfeat)], dim=1,
        ).repeat(1, model.prev_ratio, 1)
        self._prev_code_bits = prev_code_bits
        self._prev_attn_feat = prev_attn_feat
        self._audio_buffer = torch.zeros(0, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def feed(self, audio):
        if audio.dim() != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        audio = audio.to(device=self.device, dtype=self.dtype)
        self._audio_buffer = torch.cat([self._audio_buffer, audio])

        outputs = []
        while self._audio_buffer.shape[0] >= self.patch_audio_length:
            chunk = self._audio_buffer[: self.patch_audio_length]
            self._audio_buffer = self._audio_buffer[self.patch_audio_length:]
            outputs.append(self._step_chunk(chunk[None]))
        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.zeros(0, self.motion_dim, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def finish(self):
        valid_samples = self._audio_buffer.shape[0]
        if valid_samples == 0:
            return torch.zeros(0, self.motion_dim, dtype=self.dtype, device=self.device)
        valid_frames = math.ceil(valid_samples / SAMPLE_RATE * FPS)
        pad = self.patch_audio_length - valid_samples
        chunk = torch.cat([
            self._audio_buffer,
            torch.zeros(pad, dtype=self.dtype, device=self.device),
        ])
        self._audio_buffer = self._audio_buffer.new_zeros(0)
        motion = self._step_chunk(chunk[None])
        return motion[:valid_frames]

    @torch.no_grad()
    def _step_chunk(self, audio_chunk):
        model = self.model
        lvl_pos_embed = model.lvl_embed(model.lvl_idx) + model.pos_embed
        prev_lvl_pos_embed = (
            model.lvl_embed(model.lvl_idx).repeat(1, model.prev_ratio, 1)
            + model.prev_pos_embed
        )

        split_audio_feat = model.audio_encoder(audio_chunk).permute(0, 2, 1)
        split_audio_feats = [
            F.interpolate(split_audio_feat, size=(pn), mode="area").permute(0, 2, 1)
            for pn in model.patch_nums
        ]
        split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()

        next_ar_vqfeat = self._motion_style_cond
        pred_motion_bits = None
        for pidx, _pn in enumerate(model.patch_nums):
            patch_audio_cond = split_audio_cond[:, : sum(model.patch_nums[: pidx + 1])]
            patch_attn_bias = model.attn_bias_for_masking[
                :,
                :,
                : sum(model.patch_nums[: pidx + 1]),
                : sum(model.patch_nums[: pidx + 1])
                + sum(model.patch_nums) * model.prev_ratio,
            ]
            attn_feat = next_ar_vqfeat + lvl_pos_embed[:, : next_ar_vqfeat.shape[1]]
            for bidx in range(model.attn_depth):
                attn_feat = model.attn_blocks[bidx](
                    attn_feat,
                    self._prev_attn_feat + prev_lvl_pos_embed,
                    patch_audio_cond,
                    attn_bias=patch_attn_bias,
                )
            pred_motion_logits = model.logits_head(
                model.cond_logits_head(attn_feat, patch_audio_cond)
            )
            pred_motion_bits = pred_motion_logits.view(
                pred_motion_logits.shape[0], pred_motion_logits.shape[1], -1, 2,
            ).argmax(dim=-1)
            if pidx < len(model.patch_nums) - 1:
                next_ar_vqfeat = model.basic_vae.vqidx_to_ar_vqfeat(
                    pidx,
                    pred_motion_bits,
                )
                next_ar_vqfeat = torch.cat(
                    [self._motion_style_cond, model.vqfeat_embed(next_ar_vqfeat)],
                    dim=1,
                )

        _, this_pred_motion = model.basic_vae.vqidx_to_motion(
            self._prev_code_bits, pred_motion_bits
        )

        new_prev_code_bits, _ = model.basic_vae.quant_to_vqidx(
            this_pred_motion,
            this_motion=None,
        )
        new_prev_vqfeat = (
            model.basic_vae.vqidx_to_ms_vqfeat(new_prev_code_bits).detach()
        )
        this_prev_attn_feat = torch.cat(
            [self._motion_style_cond, model.vqfeat_embed(new_prev_vqfeat)], dim=1,
        )
        new_prev_attn_feat = torch.cat(
            [
                self._prev_attn_feat[:, this_prev_attn_feat.shape[1]:],
                this_prev_attn_feat,
            ],
            dim=1,
        )

        self._prev_code_bits = new_prev_code_bits
        self._prev_attn_feat = new_prev_attn_feat

        return this_pred_motion[0]


class CausalSavgolSmoother:
    """Streaming counterpart of ``ARTAvatarInferEngine.smooth_motion_savgol``.

    The one-shot smoother applies ``scipy.signal.savgol_filter`` over
    the full motion sequence in two passes:

      * all 106 channels: ``window_length=5``, ``polyorder=2``
      * pose channels (``[100:103]``) override: ``window_length=9``,
        ``polyorder=3``

    Savgol with the default ``mode='interp'`` is deterministic and
    local, so the streaming output is **bit-exact** with the one-shot
    path while filtering only a small window per feed: interior frames
    depend only on ``DELAY`` frames of context on each side, and the
    edge-fitted boundary frames are computed from slices whose edges
    coincide with the true sequence edges (the first frames are emitted
    from a slice starting at frame 0; ``finish()`` filters a slice that
    contains the final ``POSE_WINDOW`` frames). Frames that can no
    longer influence any future output are dropped, so per-feed cost
    and retained memory stay bounded regardless of session length.

    Each emitted frame is delayed by ``(POSE_WINDOW - 1) // 2 = 4``
    frames (160 ms at 25 fps), on top of the AR model's 4-second chunk
    lag.

    See ``docs/realtime.md`` for the alternative approaches that were
    considered (causal IIR/EMA, no smoothing) and why this one was
    chosen.
    """

    POSE_SLICE = slice(100, 103)
    DEFAULT_WINDOW = 5
    DEFAULT_POLY = 2
    POSE_WINDOW = 9
    POSE_POLY = 3
    DELAY = (POSE_WINDOW - 1) // 2

    def __init__(self):
        self.reset()

    def reset(self):
        # ``_tail`` holds frames [``_tail_start``, n_seen) — the suffix of
        # the full sequence that can still influence future output.
        self._tail = None
        self._tail_start = 0
        self._n_emitted = 0

    def _n_seen(self):
        return self._tail_start + self._tail.shape[0]

    def feed(self, motion_frames):
        if motion_frames.dim() != 2:
            raise ValueError(
                f"motion_frames must be 2-D (T, C), got shape {tuple(motion_frames.shape)}"
            )
        if motion_frames.shape[0] == 0:
            return motion_frames
        if self._tail is None:
            self._tail = motion_frames
        else:
            self._tail = torch.cat([self._tail, motion_frames], dim=0)
        n_seen = self._n_seen()
        empty = motion_frames.new_zeros(0, motion_frames.shape[1])
        if n_seen < self.POSE_WINDOW:
            return empty
        n_stable = n_seen - self.DELAY
        if n_stable <= self._n_emitted:
            return empty
        out = self._smooth_range(self._n_emitted, n_stable)
        self._n_emitted = n_stable
        # Future output needs DELAY frames of context before the next
        # emitted frame, and finish() needs the last POSE_WINDOW frames
        # for its right-edge polynomial fit.
        keep_from = min(self._n_emitted - self.DELAY, n_seen - self.POSE_WINDOW)
        if keep_from > self._tail_start:
            self._tail = self._tail[keep_from - self._tail_start :]
            self._tail_start = keep_from
        return out

    def finish(self):
        if self._tail is None:
            raise RuntimeError(
                "CausalSavgolSmoother.finish called before any feed; "
                "no motion_dim known."
            )
        n_seen = self._n_seen()
        if self._n_emitted >= n_seen:
            out = self._tail.new_zeros(0, self._tail.shape[1])
        elif n_seen < self.POSE_WINDOW:
            # Buffer too short to apply the 9-tap pose filter.
            # Fall back to raw frames; one-shot inference would also
            # fail on inputs this short.
            out = self._tail[self._n_emitted - self._tail_start :]
        else:
            out = self._smooth_range(self._n_emitted, n_seen)
        self._n_emitted = n_seen
        return out

    def _smooth_range(self, start, end):
        """Smoothed frames [``start``, ``end``), bit-exact with filtering
        the full sequence."""
        n_seen = self._n_seen()
        lo = max(start - self.DELAY, 0)
        if end > n_seen - self.DELAY:
            # The range includes right-boundary frames, whose edge fit
            # must see the true final POSE_WINDOW frames.
            lo = min(lo, max(n_seen - self.POSE_WINDOW, 0))
        smoothed = self._apply_savgol(self._tail[lo - self._tail_start :])
        return smoothed[start - lo : end - lo]

    @classmethod
    def _apply_savgol(cls, buffer):
        motion_np = buffer.detach().cpu().numpy()
        smoothed = savgol_filter(
            motion_np, cls.DEFAULT_WINDOW, cls.DEFAULT_POLY, axis=0,
        )
        smoothed[..., cls.POSE_SLICE] = savgol_filter(
            motion_np[..., cls.POSE_SLICE],
            cls.POSE_WINDOW, cls.POSE_POLY, axis=0,
        )
        return torch.from_numpy(smoothed).to(
            device=buffer.device,
            dtype=buffer.dtype,
        )
