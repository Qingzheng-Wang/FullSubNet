import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.nn import functional
import torchaudio as audio

from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from audio_zen.trainer.base_trainer import BaseTrainer

plt.switch_backend("agg")


class Trainer(BaseTrainer):
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
        self.model = model
        self.mel_scale = self.model.mel_scale

    def _train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in (
            tqdm(self.train_dataloader, desc="Training")
            if self.rank == 0
            else self.train_dataloader
        ):
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            noisy_mel_real = self.mel_scale(noisy_real)
            noisy_mel_imag = self.mel_scale(noisy_imag)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            clean_mel_real = self.mel_scale(clean_real)
            clean_mel_imag = self.mel_scale(clean_imag)
            cIRM_f = build_complex_ideal_ratio_mask(
                noisy_real, noisy_imag, clean_real, clean_imag
            )  # [B, F, T, 2]
            cIRM_c = build_complex_ideal_ratio_mask(
                noisy_mel_real, noisy_mel_imag, clean_mel_real, clean_mel_imag
            )  # [B, F, T, 2]
            cIRM_c = cIRM_c[:, :, :-self.model.look_ahead, :]
            cIRM_c = functional.pad(cIRM_c, [0, 0, self.model.look_ahead, 0])

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM_f, cRM_c = self.model(noisy_mag)  # [B, 2, F, T]
                cRM_f = cRM_f.permute(0, 2, 3, 1)  # [B, F, T, 2]
                cRM_c = cRM_c.permute(0, 2, 3, 1)  # [B, F, T, 2]
                loss = 0.5 * self.loss_function(cIRM_f, cRM_f) + 0.5 * self.loss_function(cIRM_c, cRM_c)

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

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            noisy_mel_real = self.mel_scale(noisy_real)
            noisy_mel_imag = self.mel_scale(noisy_imag)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            clean_mel_real = self.mel_scale(clean_real)
            clean_mel_imag = self.mel_scale(clean_imag)
            cIRM_f = build_complex_ideal_ratio_mask(
                noisy_real, noisy_imag, clean_real, clean_imag
            )  # [B, F, T, 2]
            cIRM_c = build_complex_ideal_ratio_mask(
                noisy_mel_real, noisy_mel_imag, clean_mel_real, clean_mel_imag
            )  # [B, F, T, 2]
            cIRM_c = cIRM_c[:, :, :-self.model.look_ahead, :]
            cIRM_c = functional.pad(cIRM_c, [0, 0, self.model.look_ahead, 0])

            import pdb
            pdb.set_trace()
            noisy_mag = noisy_mag.unsqueeze(1)
            cRM_f, cRM_c = self.model(noisy_mag)
            cRM_f = cRM_f.permute(0, 2, 3, 1)
            cRM_c = cRM_c.permute(0, 2, 3, 1)  # [B, F, T, 2]

            loss = 0.5 * self.loss_function(cIRM_f, cRM_f) + 0.5 * self.loss_function(cIRM_c, cRM_c)

            cRM_f = decompress_cIRM(cRM_f)

            enhanced_real = cRM_f[..., 0] * noisy_real - cRM_f[..., 1] * noisy_imag
            enhanced_imag = cRM_f[..., 1] * noisy_real + cRM_f[..., 0] * noisy_imag
            enhanced = self.torch_istft(
                (enhanced_real, enhanced_imag),
                length=noisy.size(-1),
                input_type="real_imag",
            )

            noisy = noisy.detach().squeeze(0).cpu().numpy()
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
