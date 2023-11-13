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
        self.f = config["loss_function"]["f"]

    def _train_epoch(self, epoch):
        loss_total = 0.0
        loss_f_total = 0.0
        loss_c_total = 0.0
        for noisy, clean in (
            tqdm(self.train_dataloader, desc="Training")
            if self.rank == 0
            else self.train_dataloader
        ):
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            clean_mag, clean_phase, clean_real, clean_imag = self.torch_stft(clean)
            clean_mag = clean_mag.unsqueeze(3) # [B, F, T, 1]
            cbrt_f = torch.pow(clean_mag, 1 / 3) # cubic root of clean mag as the future target
            if not self.model.look_ahead == 0:
                clean_mag = clean_mag[:, :, :-self.model.look_ahead, :]
            clean_mag = functional.pad(clean_mag, [0, 0, self.model.look_ahead, 0])
            cbrt_c = torch.pow(clean_mag, 1 / 3) # cubic root of clean mag as the current target


            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 1, F, T] => [B, F, T, 1]
                noisy_mag = noisy_mag.unsqueeze(1)
                pred_f, pred_c = self.model(noisy_mag)  # [B, 1, F, T]
                pred_f = pred_f.permute(0, 2, 3, 1)  # [B, F, T, 1]
                pred_c = pred_c.permute(0, 2, 3, 1)  # [B, F, T, 1]

                loss = self.f * self.loss_function(cbrt_f, pred_f) + (1 - self.f) * self.loss_function(cbrt_c, pred_c)
                loss_f = self.f * self.loss_function(cbrt_f, pred_f)
                loss_c = (1 - self.f) * self.loss_function(cbrt_c, pred_c)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()
            loss_f_total += loss_f.item()
            loss_c_total += loss_c.item()

        if self.rank == 0:
            self.writer.add_scalar(
                f"Loss/Train", loss_total / len(self.train_dataloader), epoch
            )
            self.writer.add_scalar(
                f"Loss/Train_f", loss_f_total / len(self.train_dataloader), epoch
            )
            self.writer.add_scalar(
                f"Loss/Train_c", loss_c_total / len(self.train_dataloader), epoch
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
            clean_mag, clean_phase, clean_real, clean_imag = self.torch_stft(clean)
            clean_mag = clean_mag.unsqueeze(3) # [B, F, T, 1]
            cbrt_f = torch.pow(clean_mag, 1 / 3) # cubic root of clean mag as the future target
            if not self.model.look_ahead == 0:
                clean_mag = clean_mag[:, :, :-self.model.look_ahead, :]
            clean_mag = functional.pad(clean_mag, [0, 0, self.model.look_ahead, 0])
            cbrt_c = torch.pow(clean_mag, 1 / 3) # cubic root of clean mag as the current target

            noisy_mag = noisy_mag.unsqueeze(1)
            pred_f, pred_c = self.model(noisy_mag)
            pred_f = pred_f.permute(0, 2, 3, 1)
            pred_c = pred_c.permute(0, 2, 3, 1)  # [B, F, T, 1]

            loss = self.f * self.loss_function(cbrt_f, pred_f) + (1 - self.f) * self.loss_function(cbrt_c, pred_c)

            pred_f = pred_f.squeeze(3)
            pred_f = torch.pow(pred_f, 3)
            pred_c = pred_c.squeeze(3)
            pred_c = torch.pow(pred_c, 3)
            pred_f_complex = pred_f * torch.exp(1j * noisy_phase)
            pred_c_complex = pred_c * torch.exp(1j * noisy_phase)

            enhanced = self.torch_istft(
                (pred_f, noisy_phase),
                length=noisy.size(-1),
                input_type="mag_phase",
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
