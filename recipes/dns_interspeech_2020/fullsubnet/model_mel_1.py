#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        model_mel_1.py
# Author:           Qingzheng WANG
# Time:             2023/9/6 22:46
# Description:                       
# Function List:    
# ===================================================

import torch
from torch.nn import functional
from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
from recipes.dns_interspeech_2020.fullsubnet.model import Model as FullSubNet

class Model(BaseModel):
    def __init__(
        self,
        num_freqs,
        look_ahead_1,
        look_ahead_2,
        sequence_model,
        fb_num_neighbors,
        sb_num_neighbors,
        fb_output_activate_function,
        sb_output_activate_function,
        fb_model_hidden_size,
        sb_model_hidden_size,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        weight_init=True,
    ):
        """FullSubNet model (cIRM mask).

        Args:
            num_freqs: frequency dim of the input
            look_ahead_1: number of use of the past frames for the first model
            look_ahead_2: number of use of the past frames for the second model
            fb_num_neighbors: number of neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: number of neighbor frequencies at each side from noisy spectrogram
            sequence_model: select one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: full-band model's activation function
            sb_output_activate_function: sub-band model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        self.FullSubNet_1 = FullSubNet(
            num_freqs,
            look_ahead_1,
            sequence_model,
            fb_num_neighbors,
            sb_num_neighbors,
            fb_output_activate_function,
            sb_output_activate_function,
            fb_model_hidden_size,
            sb_model_hidden_size,
            norm_type,
            num_groups_in_drop_band,
            weight_init)

