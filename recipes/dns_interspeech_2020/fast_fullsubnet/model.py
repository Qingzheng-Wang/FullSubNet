import torch
import torch.nn as nn
import torchaudio as audio
from torch.nn import functional, Linear

from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel

from upred import UPred


class Model(BaseModel):
    def __init__(
        self,
        look_ahead,
        shrink_size,
        sequence_model,
        num_mels,
        encoder_input_size,
        bottleneck_hidden_size,
        bottleneck_num_layers,
        noisy_input_num_neighbors,
        encoder_output_num_neighbors,
        norm_type="offline_laplace_norm",
        weight_init=False
    ):
        """Fast FullSubNet.

        Notes:
            Here, the encoder, bottleneck, and decoder are corresponding to the F_l2m, S, and F_m2l models in the paper, respectively.
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        # F_l2m
        self.encoder = nn.Sequential(
            SequenceModel(
                input_size=64,
                hidden_size=384,
                output_size=0,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None
            ),
            SequenceModel(
                input_size=384,
                hidden_size=257,
                output_size=64,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function="ReLU"
            ),
        )

        # Mel filtering
        self.mel_scale = audio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=16000,
            f_min=0,
            f_max=8000,
            n_stft=encoder_input_size,
        )

        # S
        self.bottleneck = SequenceModel(
            input_size=(noisy_input_num_neighbors * 2 + 1) + (encoder_output_num_neighbors * 2 + 1),
            output_size=1,
            hidden_size=bottleneck_hidden_size,
            num_layers=bottleneck_num_layers,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function="ReLU"
        )

        self.pred = SequenceModel(
            input_size=2 * num_mels,
            output_size=2 * num_mels,
            hidden_size=512,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function="ReLU"
        )

        # F_m2l
        self.decoder_lstm = nn.Sequential(
            SequenceModel(
                input_size=64 + 64,
                hidden_size=512,
                output_size=0,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None
            ),
            SequenceModel(
                input_size=512,
                hidden_size=512,
                output_size=257 * 1,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None,
            ),
        )

        self.full_pred = nn.Sequential(
            SequenceModel(
                input_size=64,
                hidden_size=384,
                output_size=0,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None
            ),
            SequenceModel(
                input_size=384,
                hidden_size=257,
                output_size=64,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function="ReLU"
            ),
        )

        self.sub_pred = SequenceModel(
            input_size=(noisy_input_num_neighbors * 2 + 1) + (encoder_output_num_neighbors * 2 + 1),
            output_size=1,
            hidden_size=bottleneck_hidden_size,
            num_layers=bottleneck_num_layers,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function="ReLU"
        )

        self.upred = UPred(in_channels=2, out_channels=2)

        self.shrink_size = shrink_size
        self.look_ahead = look_ahead
        self.num_mels = num_mels
        self.noisy_input_num_neighbors = noisy_input_num_neighbors
        self.enc_output_num_neighbors = encoder_output_num_neighbors
        self.norm = self.norm_wrapper(norm_type)
        self.encoder_input_size = encoder_input_size

        # predict the future frames according to the enhanced current frames
        # c2f is current to future
        self.linear_c2f = Linear(6 * 64 , 2 * 64)

        if weight_init:
            self.apply(self.weight_init)

    def real_time_downsampling(self, input):
        """Downsampling an input tensor long time.

        Args:
            input: tensor with the shape of [B, C, F, T].

        Returns:
            Donwsampled tensor with the shape of [B, C, F, T // shrink_size].
        """
        first_block = input[..., 0:1]  # [B, C, F, 1]
        block_list = torch.split(input[..., 1:], self.shrink_size, dim=-1)  # ([B, C, F, shrink_size], [B, C, F, shrink_size], ...)
        last_block = block_list[-1]  # [B, C, F, shrink_size]

        output = torch.cat(
            (
                first_block,
                torch.mean(torch.stack(block_list[:-1], dim=-1), dim=-2),
                torch.mean(last_block, dim=-1, keepdim=True)
            ), dim=-1
        )  # [B, C, F, T // shrink_size]

        return output

    def real_time_upsampling(self, input, target_len=False):
        *_, n_frames = input.shape
        input = input[..., None]  # [B, C, F, T, 1]
        input = input.expand(*input.shape[:-1], self.shrink_size)  # [B, C, F, T, shrink_size]
        input = input.reshape(*input.shape[:-2], n_frames * self.shrink_size)  # [B, C, F, T * shrink_size]

        if target_len:
            input = input[..., :target_len]

        return input

    def mel_rescale(self, pred_mel):
        """Rescale the predicted mel spectrogram to suppress the weak background noise.

        Args:
            pred_mel: predicted mel spectrogram

        Returns:
            Rescaled mel spectrogram
        """
        # create a tensor with shape same as pred_mel and fill it with 1 * e-4
        threshold = torch.ones_like(pred_mel) * 1e-4
        pred_mel = torch.max(pred_mel, threshold)
        pred_mel = 20 * torch.log10(pred_mel) - 20
        pred_mel = (pred_mel + 100) / 100
        # if pred_mel < 0, pred_mel = 0,
        # else if 0 <= pred_me pred_mel <= 1 pred_mel = pred_mel,
        # else pred_mel = 1
        pred_mel = torch.clamp(pred_mel, min=0, max=1)

        return  pred_mel

    # fmt: off
    def forward(self, mix_mag):
        """Forward pass.

        Args:
            mix_mag: noisy magnitude spectrogram with shape [B, 1, F, T].

        Returns:
            The real part and imag part of the enhanced spectrogram with shape [B, 2, F, T].

        Notes:
            B - batch size
            C - channel
            F - frequency
            F_mel - mel frequency
            T - time
            F_s - sub-band frequency
        """
        assert mix_mag.dim() == 4
        mix_mag = functional.pad(mix_mag, [self.look_ahead, 0])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = mix_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes a magnitude feature as the input."

        # Mel filtering
        mix_pow = torch.pow(mix_mag, 2)
        mix_mel_pow = self.mel_scale(mix_pow)  # [B, C, F_mel, T]
        _, _, num_freqs_mel, _ = mix_mel_pow.shape
        mix_mel_pow = torch.pow(mix_mel_pow, 1 / 3)

        # F_l2m
        enc_input = self.norm(mix_mel_pow).reshape(batch_size, -1, num_frames)
        enc_output = self.encoder(enc_input).reshape(batch_size, num_channels, -1, num_frames)  # [B, C, F_mel, T]

        # Unfold - noisy spectrogram, [B, N=F, C, F_s, T]
        mix_mel_pow_unfold = self.freq_unfold(mix_mel_pow, num_neighbors=self.noisy_input_num_neighbors)  # [B, F_mel, C, F_sub, T]
        mix_mel_pow_unfold = mix_mel_pow_unfold.reshape(batch_size, self.num_mels, self.noisy_input_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Unfold - full-band model's output, [B, N=F, C, F_f, T], where N is the number of sub-band units
        enc_output_unfold_mel = self.freq_unfold(enc_output, num_neighbors=self.enc_output_num_neighbors)  # [B, F_mel, C, F_sub, T]
        enc_output_unfold_mel = enc_output_unfold_mel.reshape(batch_size, self.num_mels, self.enc_output_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Bottleneck (S)
        bn_input = torch.cat([mix_mel_pow_unfold, enc_output_unfold_mel], dim=2)
        num_sb_unit_freqs = bn_input.shape[2]

        bn_input = bn_input.reshape(batch_size * self.num_mels, num_sb_unit_freqs, num_frames)  # [B * F_mel, F_sub_1 + F_sub_2, T]
        bn_output = self.bottleneck(bn_input)  # [B * F_mel, 1, T]
        bn_output = bn_output.reshape(batch_size, self.num_mels, 1, num_frames).permute(0, 2, 1, 3)  # [B, 1, F_mel, T]

        # output predicted current mel spectrogram
        pred_mel_c = bn_output.permute(0, 2, 3, 1) # [B, F_mel, T, 1]
        bn_output_prev = bn_output
        pred_mel_c = pred_mel_c[:, :, :num_frames-self.look_ahead, :]

        # Fuse full-band and sub-band model output
        bn_output = torch.cat([bn_output, enc_output], dim=1)  # [B, 2, F_mel, T]

        output_c = bn_output
        output_c = self.decoder_lstm(output_c.reshape(batch_size, 2 * num_freqs_mel, num_frames))  # [B, F_mel * 1, T]
        output_c = output_c.reshape(batch_size, 1, num_freqs, num_frames)
        output_c = output_c[:, :, :, :num_frames-self.look_ahead]

        # ==========current to future==========
        # bn_output = self.pred(bn_output.reshape(batch_size, -1, num_frames))  # [B, 2 * F_mel, T]

        bn_output = bn_output.permute(0, 1, 3, 2)
        bn_output = self.upred(bn_output).permute(0, 1, 3, 2) # [B, 2, F_mel, T]
        # # ----------pred_full_band----------
        # full_pred_input = bn_output_prev.permute(0, 3, 1, 2) # [B, 1, F_mel, T]
        # full_pred_input = full_pred_input.reshape(batch_size, -1, num_frames) # [B, F_mel, T]
        # full_pred_output = self.full_pred(full_pred_input).reshape(batch_size, num_channels, -1, num_frames)
        # # ----------pred_sub_band----------
        # full_pred_input = full_pred_input.reshape(batch_size, 1, num_freqs_mel, num_frames) # [B, 1, F_mel, T]
        # full_pred_input_unfold = self.freq_unfold(full_pred_input, num_neighbors=self.noisy_input_num_neighbors)  # [B, F_mel, C, F_sub, T]
        # full_pred_input_unfold = full_pred_input_unfold.reshape(batch_size, self.num_mels, self.noisy_input_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]
        # full_pred_input = full_pred_input.reshape(batch_size, num_freqs_mel, 1, num_frames)
        # sub_pred_input = torch.cat([full_pred_input_unfold, full_pred_input], dim=2)
        # sub_pred_input = sub_pred_input.reshape(batch_size * self.num_mels, -1, num_frames)  # [B * F_mel, F_sub, T]
        # sub_pred_output = self.sub_pred(sub_pred_input).reshape(batch_size, self.num_mels, -1, num_frames).permute(0, 2, 1, 3)  # [B, 1, F_mel, T]
        # # ----------Fm2l-------------------
        # f_m2l_input = torch.cat([full_pred_output, sub_pred_output], dim=1)
        bn_output = bn_output.reshape(batch_size, 2, num_freqs_mel, num_frames)  # [B, 2, F_mel, T]

        # F_ml2
        dec_input = bn_output.reshape(batch_size, -1, num_frames)  # [B, 2 * F_mel, T]
        # dec_input = self.mel_rescale(dec_input) # rescale the predicted mel spectrogram to suppress the weak background noise
        decoder_lstm_output = self.decoder_lstm(dec_input)  # [B, F * 1, T]
        dec_output = decoder_lstm_output.reshape(batch_size, 1, num_freqs, num_frames)

        # Output
        output_f = dec_output[:, :, :, :num_frames-self.look_ahead]

        return output_f, output_c, pred_mel_c # predicted future frame, enhanced current frame

# fmt: on
if __name__ == "__main__":
    import time

    with torch.no_grad():
        noisy_mag = torch.rand(1, 1, 257, 63)
        model = Model(
            look_ahead=2,
            shrink_size=2,
            sequence_model="LSTM",
            num_mels=64,
            encoder_input_size=257,
            bottleneck_hidden_size=384,
            bottleneck_num_layers=2,
            noisy_input_num_neighbors=5,
            encoder_output_num_neighbors=0
        )
        start = time.time()
        output = model(noisy_mag)
        end = time.time()
        print(end - start)
        # summary(model, (1, 1, 257, 63), device="cpu")
