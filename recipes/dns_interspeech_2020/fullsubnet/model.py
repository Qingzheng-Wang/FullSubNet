import torch
from torch.nn import functional
from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel


class Model(BaseModel):
    def __init__(
        self,
        num_freqs,
        look_ahead,
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
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in (
            "GRU",
            "LSTM",
        ), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function,
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        # LH is the abbreviation of look ahead!!!
        # [B, C, F, T] => [B, C, F, T + LH]
        noisy_mag = functional.pad(noisy_mag, [self.look_ahead, 0])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert (
            num_channels == 1
        ), f"{self.__class__.__name__} takes the mag feature as inputs."

        # Full-band model
        # [B, C, F, T + LH] => [B, C * F, T + LH]
        fb_input = self.norm(noisy_mag).reshape(
            batch_size, num_channels * num_freqs, num_frames
        )

        # [B, C * F, T + LH]
        # => INNER{ [B, T + LH, C * F] -> [B, T + LH, C * F] -> [B, C * F, T + LH]}
        # => [B, C, F, T + LH]
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output
        # [B, C, F, T + LH] => [B, N=F, C, F_f(1), T + LH]
        fb_output_unfolded = self.freq_unfold(fb_output, num_neighbors=self.fb_num_neighbors)
        # [B, N=F, C, F_f(1), T + LH] => [B, N=F, F_f(1), T + LH]
        fb_output_unfolded = fb_output_unfolded.reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames
        )

        # Unfold noisy spectrogram
        # [B, C, F, T + LH] => [B, N=F, C, F_s(2 * neighbors + 1), T + LH]
        noisy_mag_unfolded = self.freq_unfold(noisy_mag, num_neighbors=self.sb_num_neighbors)
        # [B, N=F, C, F_s(2 * neighbors + 1), T + LH] => [B, N=F, F_s(2 * neighbors + 1), T + LH]
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(
            batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames
        )
        # Concatenation
        # [B, N = F, (F_s + F_f), T + LH]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation.
        # 降维，将频率维度分组，每组的频率数为F / num_groups，目的是减少计算量
        if batch_size > 1:
            sb_input = drop_band(
                sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band
            )  # [B, (F_s + F_f), F / num_groups, T + LH]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F / num_groups, (F_s + F_f), T + LH]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames,
        ) # [B * F / num_groups, (F_s + F_f), T + LH]

        # [B * F / num_groups, (F_s + F_f), T +LH]
        # => [B * F / num_groups, 1, T + LH]
        # => [B, 1, F / num_groups, T + LH]
        sb_mask = self.sb_model(sb_input)
        sb_mask = (
            sb_mask.reshape(batch_size, num_freqs, 2, num_frames)
            .permute(0, 2, 1, 3)
            .contiguous() # 这里的contiguous()是为了解决内存不连续的问题
        )
        output = sb_mask[:, :, :, :num_frames-self.look_ahead] # [B, 2, F / num_groups, T]
        return output


if __name__ == "__main__":
    with torch.no_grad():
        noisy_mag = torch.rand(1, 1, 257, 63)
        model = Model(
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        print(model(noisy_mag).shape)
