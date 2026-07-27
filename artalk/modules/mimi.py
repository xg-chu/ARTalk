import torch
import torchaudio
import torch.nn as nn
from transformers import MimiModel

class MimiModelWrapper(nn.Module):
    def __init__(self, model_name="kyutai/mimi"):
        super().__init__()
        self.acoustic_model = MimiModel.from_pretrained(model_name).eval()
        self.latent_dim = self.acoustic_model.config.hidden_size
        for p in self.acoustic_model.parameters():
            p.requires_grad = False

    def forward(self, audio):
        audio = torchaudio.functional.resample(audio, orig_freq=16000, new_freq=24000)
        codes = self.acoustic_model.encode(audio[:, None]).audio_codes
        features = self.acoustic_model.quantizer.decode(codes)
        # features = self.acoustic_model.upsample(features)
        return features.permute(0, 2, 1)
