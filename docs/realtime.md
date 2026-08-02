# Realtime streaming notes

Design notes for the streaming-friendly extension of ARTalk inference
(roadmap: [#30](https://github.com/xg-chu/ARTalk/issues/30)).

## Constraints

The released model processes audio in fixed 4-second / 100-frame chunks.
Multi-scale autoregressive decoding within each chunk depends on the
full chunk being available, so end-to-end latency cannot drop below
~4 seconds without architectural changes to the AR model (smaller
`patch_nums`, causal redesign, etc.). All streaming work preserves the
released checkpoints bit-for-bit unchanged; latency reductions below
the 4-second floor would require coordinated retraining work.

## Streaming inference — `ARTalkStreamer`

Refactors `BitwiseARModel.inference`'s chunked audio loop into a
stateful feed/finish API in `artalk/streaming.py`. The model itself and
the existing one-shot inference path are untouched, so the released
checkpoint and the CLI / Gradio app continue to work unchanged.

`scripts/check_streaming_parity.py` asserts the streaming output is
bit-exact with the one-shot path. Verified on the released wav2vec
checkpoint: `max abs diff: 0.000e+00`.

## Causal motion smoother — `CausalSavgolSmoother`

Streaming-friendly equivalent of the post-processing in
`ARTAvatarInferEngine.smooth_motion_savgol` (scipy `savgol_filter` over
the full motion sequence — non-causal).

### Design decisions

Three approaches were considered:

- **Delayed savgol.** Buffer future motion frames and emit each
  output with `(POSE_WINDOW - 1) // 2 = 4` frames (160 ms at 25 fps)
  of additional latency. Re-running scipy's `savgol_filter` on the
  growing buffer yields output that is **bit-exact** with the
  one-shot path: `mode='interp'` (the scipy default) is deterministic,
  and interior frames depend only on a fixed `window`-sized
  neighborhood that becomes context-independent once both halves of
  the window are in the buffer.
- **Causal IIR / EMA.** One-pole low-pass filter applied to past
  samples only. Zero added latency, but the frequency response
  differs from savgol; output values would not match the one-shot
  reference.
- **No smoothing.** The original smoother is cosmetic; some use
  cases may not need it.

**Choice: delayed savgol.** The 160 ms added latency is small relative
to the 4-second chunk floor, and bit-exact output keeps the smoother
from being a confounding variable when verifying the later streaming
stages. "No smoothing" remains available as a simple opt-out at the
call site (do not run motion through the smoother). The causal IIR
option is recorded for reference; it would be considered together with
retraining work since it would not be bit-exact regardless.

Other engine post-processing (eye-channel zeroing, `fix_pose`,
`clip_length` truncation) is not streaming-aware and stays at the
engine layer.

## Planned follow-ups

Per the [roadmap](https://github.com/xg-chu/ARTalk/issues/30): a
per-frame streaming renderer mirroring the mesh / GAGAvatar branches of
`ARTAvatarInferEngine.rendering`, and a realtime pipeline that wires
`ARTalkStreamer → CausalSavgolSmoother → renderer` behind
audio-clock-synchronized output callbacks. A WebRTC application layer
built on these pieces lives in
[artalk-streamlit-realtime](https://github.com/whitphx/artalk-streamlit-realtime).

## Future (deferred)

- **Architectural latency reduction below the 4-second floor.** Smaller
  `patch_nums`, causal AR redesign, etc. Requires retraining and
  coordination with the original author.
- **Client-side rendering.** Push raw 106-dim motion params over the
  wire instead of rendered video and implement FLAME / GAGAvatar
  equivalents in the browser (Three.js + Gaussian splat). Drastically
  reduces server load and bandwidth, and decouples the rendering
  pipeline from the inference server.
