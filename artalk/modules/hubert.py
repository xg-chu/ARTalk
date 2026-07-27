#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
from transformers import HubertModel

class HubertModel(HubertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_values, frame_num=None):
        input_values = self.normalize_audio(input_values)
        extract_features = self.feature_extractor(input_values)  # (N, C, L)
        if frame_num is not None:
            extract_features = torch.nn.functional.interpolate(
                extract_features, size=frame_num, align_corners=False, mode='linear'
            )
        extract_features = extract_features.transpose(1, 2)  # (N, L, C)
        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states)
        encoder_outputs = self.encoder(
            hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True
        )
        hidden_states = encoder_outputs[0]
        return hidden_states

    @staticmethod
    def normalize_audio(audio_tensor, dim=-1):
        audio_mean = audio_tensor.mean(dim=dim, keepdim=True)
        audio_std = audio_tensor.std(dim=dim, keepdim=True)
        audio_tensor = (audio_tensor - audio_mean) / (audio_std + 1e-6)
        return audio_tensor
