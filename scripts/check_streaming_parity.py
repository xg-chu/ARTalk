#!/usr/bin/env python
"""Parity checks for the streaming inference / post-processing pieces.

Loads the released ARTalk wav2vec checkpoint and runs:

  1. **Streaming inference** — feeds the same audio to
     ``BitwiseARModel.inference`` (one-shot) and to ``ARTalkStreamer``
     (chunked, arbitrary feed sizes), and asserts the motion outputs
     match within ``--atol``.

  2. **Streaming smoother** — applies
     ``ARTAvatarInferEngine.smooth_motion_savgol`` (one-shot) and
     ``CausalSavgolSmoother`` (streaming) to the same motion sequence,
     and asserts the outputs match within ``--atol``.

Run on a GPU host where the model and ``./assets`` are available:

    python -m scripts.check_streaming_parity [-a demo/eng1.wav] [--device cuda]

(or ``PYTHONPATH=. python scripts/check_streaming_parity.py ...``)
"""

import argparse
import json

import torch
import torchaudio
from scipy.signal import savgol_filter

from artalk import BitwiseARModel
from artalk.streaming import ARTalkStreamer, CausalSavgolSmoother


def smooth_motion_savgol_reference(motion_codes):
    """One-shot reference: mirror of ``ARTAvatarInferEngine.smooth_motion_savgol``."""
    motion_np = motion_codes.clone().detach().cpu().numpy()
    motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
    motion_np_smoothed[..., 100:103] = savgol_filter(
        motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
    )
    return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def run_streamer(streamer, audio, feed_chunk_samples):
    pieces = []
    for i in range(0, audio.shape[0], feed_chunk_samples):
        out = streamer.feed(audio[i : i + feed_chunk_samples])
        if out.shape[0] > 0:
            pieces.append(out)
    tail = streamer.finish()
    if tail.shape[0] > 0:
        pieces.append(tail)
    return torch.cat(pieces, dim=0) if pieces else None


def run_smoother(smoother, motion, feed_chunk_frames):
    pieces = []
    for i in range(0, motion.shape[0], feed_chunk_frames):
        out = smoother.feed(motion[i : i + feed_chunk_frames])
        if out.shape[0] > 0:
            pieces.append(out)
    tail = smoother.finish()
    if tail.shape[0] > 0:
        pieces.append(tail)
    return torch.cat(pieces, dim=0) if pieces else None


def assert_match(reference, streamed, atol, label):
    print(f"[{label}] one_shot: {tuple(reference.shape)}, streamed: {tuple(streamed.shape)}")
    assert reference.shape == streamed.shape, f"[{label}] shape mismatch"
    diff = (reference - streamed).abs()
    print(f"[{label}] max abs diff:  {diff.max().item():.3e}")
    print(f"[{label}] mean abs diff: {diff.mean().item():.3e}")
    if not torch.allclose(reference, streamed, atol=atol):
        raise SystemExit(f"[{label}] diverges beyond atol={atol}")
    print(f"[{label}] OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", default="./demo/eng1.wav", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--feed-chunk-samples",
        type=int,
        default=4000,
        help="audio chunk size fed to ARTalkStreamer (samples @16kHz). "
             "Choose != patch_audio_length so audio buffering is exercised.",
    )
    parser.add_argument(
        "--smoother-feed-chunk-frames",
        type=int,
        default=7,
        help="motion chunk size fed to CausalSavgolSmoother (frames @25fps). "
             "Choose a small odd number to exercise sub-window feeds.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--audio-encoder", default="wav2vec", type=str,
        help="ARTalk audio encoder architecture name.",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str,
        help="checkpoint path override (default: ./assets/ARTalk_<audio-encoder>.pt)",
    )
    parser.add_argument(
        "--style", default=None, type=str,
        help="optional style id under assets/style_motion (e.g. natural_0)",
    )
    args = parser.parse_args()

    device = args.device
    audio_encoder = args.audio_encoder

    checkpoint_path = args.checkpoint or f"./assets/ARTalk_{audio_encoder}.pt"
    print(f"checkpoint: {checkpoint_path} (audio encoder: {audio_encoder})")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configs = json.load(open("./assets/config.json"))
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
    model = BitwiseARModel(configs).to(device)
    model.train(False)
    model.load_state_dict(ckpt, strict=True)

    style_motion = None
    if args.style is not None:
        style_motion = torch.load(
            f"./assets/style_motion/{args.style}.pt", map_location="cpu", weights_only=True
        )

    audio, sr = torchaudio.load(args.audio)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0).to(device)
    print(f"audio: {audio.shape[0]} samples ({audio.shape[0] / 16000:.2f}s)")

    batch = {"audio": audio[None]}
    if style_motion is not None:
        batch["style_motion"] = style_motion[None].to(device)
    one_shot_motion = model.inference(batch)[0]

    # Streaming inference parity.
    streamer = ARTalkStreamer(model, style_motion=style_motion)
    streamed_motion = run_streamer(streamer, audio, args.feed_chunk_samples)
    assert streamed_motion is not None, "streamer produced no output"
    assert_match(one_shot_motion, streamed_motion, args.atol, "streaming inference")

    # Streaming smoother parity (against one-shot smoother).
    one_shot_smoothed = smooth_motion_savgol_reference(one_shot_motion)
    smoother = CausalSavgolSmoother()
    streamed_smoothed = run_smoother(
        smoother, one_shot_motion, args.smoother_feed_chunk_frames
    )
    assert streamed_smoothed is not None, "smoother produced no output"
    assert_match(one_shot_smoothed, streamed_smoothed, args.atol, "streaming smoother")


if __name__ == "__main__":
    main()
