#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        trainer_mel.py
# Author:           Qingzheng WANG
# Time:             2023/9/3 21:06
# Description:                       
# Function List:    
# ===================================================

import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from audio_zen.acoustics.feature import drop_band
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from audio_zen.trainer.base_trainer import BaseTrainer

plt.switch_backend("agg")


class TrainerMel(BaseTrainer):
    def __init__(
        self,
        dist,
        rank,
        config,
        resume,
        only_validation,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        validation_dataloader,
    ):
        super().__init__(
            dist, rank, config, resume, only_validation, model, loss_function, optimizer
        )
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy, clean in (
            tqdm(self.train_dataloader, desc="Training")
            if self.rank == 0
            else self.train_dataloader
        ):
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank) # [B, T]
            clean = clean.to(self.rank)

            noisy_mel = self.torch_mel(noisy) # [B, F, T], F = n_mels
            clean_mel = self.torch_mel(clean)
            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 1, F, T] => [B, F, T]
                noisy_mel = noisy_mel.unsqueeze(1)
                o = self.model(noisy_mel)
                o = o.squeeze(1)
                loss = self.loss_function(clean_mel, o)

            # 缩放浮点数精度，减少显存占用
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        if self.rank == 0:
            self.writer.add_scalar(
                f"Loss/Train", loss_total / len(self.train_dataloader), epoch
            )

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {
            "With_reverb": 0.0,
            "No_reverb": 0.0,
        }
        item_idx_list = {
            "With_reverb": 0,
            "No_reverb": 0,
        }
        noisy_y_list = {
            "With_reverb": [],
            "No_reverb": [],
        }
        clean_y_list = {
            "With_reverb": [],
            "No_reverb": [],
        }
        enhanced_y_list = {
            "With_reverb": [],
            "No_reverb": [],
        }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, name, speech_type) in tqdm(
            enumerate(self.valid_dataloader), desc="Validation"
        ):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            _, noisy_phase, _, _ = self.torch_stft(noisy)
            noisy_mel = self.torch_mel(noisy)  # [B, F, T], F = n_mels
            clean_mel = self.torch_mel(clean)
            noisy_mel = noisy_mel.unsqueeze(1)
            o = self.model(noisy_mel)
            o = o.squeeze(1)

            loss = self.loss_function(clean_mel, o)

            enhanced = self.torch_imel(o, noisy_phase=noisy_phase) # reconstructed wav, [B, T]

            noisy = noisy.detach().squeeze(0).cpu().numpy() # [B * T]
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            if item_idx_list[speech_type] <= visualization_n_samples:
                self.spec_audio_visualization(
                    noisy, enhanced, clean, name, epoch, mark=speech_type
                )

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

        self.writer.add_scalar(
            f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch
        )

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(
                f"Loss/{speech_type}",
                loss_list[speech_type] / len(self.valid_dataloader),
                epoch,
            )

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type],
                clean_y_list[speech_type],
                enhanced_y_list[speech_type],
                visualization_metrics,
                epoch,
                visualization_num_workers,
                mark=speech_type,
            )

        return validation_score_list["With_reverb"]
