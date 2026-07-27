#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_values):
        input_values = self.normalize_audio(input_values)
        hidden_states = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states = self.feature_projection(hidden_states)[0]
        encoder_outputs = self.encoder(
            hidden_states, attention_mask=None, output_attentions=False,
            output_hidden_states=False, return_dict=True,
        )
        hidden_states = encoder_outputs[0]
        return hidden_states

    @staticmethod
    def normalize_audio(audio_tensor, dim=-1):
        audio_mean = audio_tensor.mean(dim=dim, keepdim=True)
        audio_std = audio_tensor.std(dim=dim, keepdim=True)
        audio_tensor = (audio_tensor - audio_mean) / (audio_std + 1e-6)
        return audio_tensor
