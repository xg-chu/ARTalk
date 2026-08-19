#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Public ARTalk runtime API for external integrations.

This module keeps ARTalk inference importable by external programs without
requiring them to use the demo scripts. It intentionally returns structured
Python objects; application-specific artifact writing belongs in caller code.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from scipy.signal import savgol_filter

from artalk.assets import ARTalkAssets
from artalk.flame_model.FLAME import FLAMEModel
from artalk.models import BitwiseARModel


MESH_REGION_LABELS = {
    "skin": 0,
    "lips": 1,
    "mouth": 2,
    "eye": 3,
}
MEDIAPIPE_EYE_LANDMARKS = {
    7,
    33,
    133,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
    466,
}
MEDIAPIPE_LIP_LANDMARKS = {
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
}
MEDIAPIPE_MOUTH_LANDMARKS = {
    13,
    14,
    17,
    78,
    80,
    81,
    82,
    87,
    88,
    95,
    178,
    191,
    308,
    310,
    311,
    312,
    317,
    318,
    321,
    324,
    402,
    415,
}


@dataclass(frozen=True)
class ARTalkRuntimeConfig:
    asset_dir: Path | str | None = None
    assets: ARTalkAssets | None = None
    audio_encoder: str = "wav2vec"
    # Explicit checkpoint override for model variants sharing an audio
    # encoder (e.g. retrained weights); defaults to the asset tree's
    # ARTalk_<audio_encoder>.pt.
    checkpoint_path: Path | str | None = None
    device: str = "auto"
    clip_length: int = 750
    fps: int = 25
    sample_rate: int = 16000
    fix_pose: bool = False
    flame_scale: float = 1.0

    def resolved_assets(self) -> ARTalkAssets:
        if self.assets is not None:
            return self.assets
        return ARTalkAssets.resolve(root=self.asset_dir)

    def resolved_asset_dir(self) -> Path:
        return self.resolved_assets().root

    def resolved_checkpoint(self) -> Path:
        if self.checkpoint_path is not None:
            return Path(self.checkpoint_path).expanduser()
        return self.resolved_assets().checkpoint(self.audio_encoder)


@dataclass
class ARTalkResult:
    audio: torch.Tensor
    motions: torch.Tensor
    vertices: np.ndarray
    faces: np.ndarray
    region_labels: np.ndarray
    region_source: str
    sample_rate: int
    fps: int
    avatar_id: str


class ARTalkRuntime:
    """Reusable ARTalk motion and FLAME mesh generator."""

    def __init__(self, config: ARTalkRuntimeConfig | None = None):
        self.config = config or ARTalkRuntimeConfig()
        self.assets = self.config.resolved_assets()
        self.checkpoint_path = self.config.resolved_checkpoint()
        self.assets.validate(
            self.config.audio_encoder, checkpoint_path=self.checkpoint_path
        )
        self.asset_dir = self.assets.root
        self.device = select_device(self.config.device)
        ckpt = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        with self.assets.config.open() as f:
            configs = json.load(f)
        configs["AR_CONFIG"]["AUDIO_ENCODER"] = self.config.audio_encoder
        self.model = BitwiseARModel(configs).eval().to(self.device)
        self.model.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(
            n_shape=300,
            n_exp=100,
            scale=self.config.flame_scale,
            no_lmks=True,
            model_path=self.assets.flame_model,
        ).to(self.device)
        self.style_motion = None

    def available_styles(self) -> list[str]:
        return available_styles(self.asset_dir)

    def set_style_motion(self, style_id: str | None):
        if style_id in (None, "", "default"):
            self.style_motion = None
            return
        style_motion = torch.load(
            self.asset_dir / "style_motion" / f"{style_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if tuple(style_motion.shape) != (50, 106):
            raise ValueError(f"Invalid style motion shape for {style_id}: {style_motion.shape}")
        self.style_motion = style_motion[None].to(self.device)

    @torch.no_grad()
    def generate(
        self,
        audio_path: str | Path,
        *,
        style_id: str = "default",
        clip_length: int | None = None,
        avatar_id: str = "mesh",
        shape_code: torch.Tensor | None = None,
    ) -> ARTalkResult:
        audio, sr = load_audio(audio_path)
        audio = torchaudio.transforms.Resample(sr, self.config.sample_rate)(audio).mean(dim=0)
        self.set_style_motion(style_id)
        audio_batch = {
            "audio": audio[None].to(self.device),
            "style_motion": self.style_motion,
        }
        pred_motions = self.model.inference(audio_batch, with_gtmotion=False)[0]
        limit = clip_length if clip_length is not None else self.config.clip_length
        pred_motions = smooth_motion_savgol(pred_motions)[:limit]
        if self.config.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        pred_motions[..., 104:] *= 0.0

        if shape_code is None:
            shape_code = audio.new_zeros(1, 300)
        elif shape_code.dim() == 1:
            shape_code = shape_code[None]
        if tuple(shape_code.shape) != (1, 300):
            raise ValueError(f"shape_code must be (300,) or (1, 300), got {tuple(shape_code.shape)}")
        shape_code = shape_code.to(self.device).expand(pred_motions.shape[0], -1)
        vertices = self.model.basic_vae.get_flame_verts(
            self.flame_model,
            shape_code,
            pred_motions,
            with_global=True,
        )
        audio = audio[: int(vertices.shape[0] / self.config.fps * self.config.sample_rate)]
        faces = self.flame_model.get_faces().cpu().numpy().astype(np.int32, copy=False)
        return ARTalkResult(
            audio=audio.float().cpu(),
            motions=pred_motions.float().cpu(),
            vertices=vertices.float().cpu().numpy().astype(np.float32, copy=False),
            faces=faces,
            region_labels=build_mesh_region_labels(
                int(vertices.shape[1]),
                faces,
                mediapipe_region_seed_faces(self.flame_model),
            ),
            region_source="mediapipe-landmark-adjacency-v1",
            sample_rate=self.config.sample_rate,
            fps=self.config.fps,
            avatar_id=avatar_id,
        )


def available_styles(asset_dir: str | Path = "assets") -> list[str]:
    style_dir = Path(asset_dir) / "style_motion"
    if not style_dir.exists():
        return []
    return sorted(path.stem for path in style_dir.glob("*.pt"))


def select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cpu")


def smooth_motion_savgol(motion_codes: torch.Tensor) -> torch.Tensor:
    motion_np = motion_codes.clone().detach().cpu().numpy()
    motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
    motion_np_smoothed[..., 100:103] = savgol_filter(
        motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
    )
    return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def load_audio(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(audio_path)
    try:
        return torchaudio.load(str(path))
    except RuntimeError as exc:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "Could not decode audio with torchaudio. Install an audio backend "
                "such as ffmpeg/soundfile, or provide a WAV file."
            ) from exc
        try:
            sr, data = wavfile.read(path)
        except Exception as wav_exc:
            raise exc from wav_exc
        if data.ndim == 1:
            data = data[None, :]
        else:
            data = data.T
        audio_np = data.astype(np.float32, copy=False)
        if np.issubdtype(data.dtype, np.unsignedinteger):
            info = np.iinfo(data.dtype)
            midpoint = float(info.max + 1) / 2.0
            audio_np = (audio_np - midpoint) / midpoint
        elif np.issubdtype(data.dtype, np.signedinteger):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), info.max))
            audio_np = audio_np / scale
        return torch.from_numpy(audio_np), sr


def save_audio(audio_path: str | Path, audio: torch.Tensor, sample_rate: int):
    path = Path(audio_path)
    try:
        torchaudio.save(str(path), audio, sample_rate)
        return
    except RuntimeError as exc:
        if path.suffix.lower() != ".wav":
            raise exc
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    audio_np = np.clip(audio_np, -1.0, 1.0)
    wavfile.write(path, sample_rate, (audio_np * 32767.0).astype(np.int16))


def build_mesh_region_labels(vertex_count: int, faces: np.ndarray, region_seed_faces: dict) -> np.ndarray:
    labels = np.full(vertex_count, MESH_REGION_LABELS["skin"], dtype=np.uint8)
    neighbors = build_vertex_neighbors(vertex_count, faces)
    for name, depth in (("lips", 2), ("eye", 1), ("mouth", 1)):
        seed_faces = region_seed_faces.get(name)
        if seed_faces is None or len(seed_faces) == 0:
            continue
        vertices = grow_region_vertices(faces[seed_faces].reshape(-1), neighbors, depth)
        labels[vertices] = MESH_REGION_LABELS[name]
    return labels


def build_vertex_neighbors(vertex_count: int, faces: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def grow_region_vertices(seed_vertices: np.ndarray, neighbors: list[set[int]], depth: int) -> np.ndarray:
    region = set(int(vertex) for vertex in seed_vertices)
    frontier = set(region)
    for _ in range(depth):
        next_frontier = set()
        for vertex in frontier:
            next_frontier.update(neighbors[vertex])
        next_frontier.difference_update(region)
        region.update(next_frontier)
        frontier = next_frontier
    return np.fromiter(region, dtype=np.int64)


def mediapipe_region_seed_faces(flame_model: FLAMEModel) -> dict[str, np.ndarray]:
    ckpt = flame_model.flame_ckpt["lmk_embeddings_mediapipe"]
    landmark_ids = ckpt["landmark_indices"].detach().cpu().numpy()
    landmark_face_indices = flame_model.lmk_faces_idx_mediapipe.detach().cpu().numpy()
    return {
        "lips": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_LIP_LANDMARKS),
        "mouth": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_MOUTH_LANDMARKS),
        "eye": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_EYE_LANDMARKS),
    }


def landmark_faces_for_ids(
    landmark_ids: np.ndarray,
    landmark_face_indices: np.ndarray,
    selected_ids: set[int],
) -> np.ndarray:
    mask = np.isin(landmark_ids, list(selected_ids))
    return np.unique(landmark_face_indices[mask]).astype(np.int64, copy=False)
