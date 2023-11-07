import os

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim, InverseSpectrogram, Spectrogram, MelScale
from torchaudio import functional as F

def stft(y, n_fft, hop_length, win_length):
    """Wrapper of the official torch.stft for single-channel and multichannel.

    Args:
        y: single- or multichannel speech with shape of [B, C, T] or [B, T]
        n_fft: number of FFT
        hop_length: hop length
        win_length: hanning window size

    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]

    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)  # [B * C ,T]

    complex_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

    mag = torch.abs(complex_stft)
    phase = torch.angle(complex_stft)
    real = complex_stft.real
    imag = complex_stft.imag
    return mag, phase, real, imag

def mel(y, sr, n_fft, hop_length, win_length, n_mels, f_min=0., f_max=None):

    # spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length,
    #                           win_length=win_length, window_fn=torch.hann_window, wkwargs={'device':y.device})
    # spectrogram = spectrogram(y)
    # fb = F.melscale_fbanks(n_fft//2 + 1, f_min=0., f_max=8000., n_mels=n_mels, sample_rate=sr).to(y.device)
    # mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), fb).transpose(-1, -2)
    # return mel_spectrogram

    transform = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, n_mels=n_mels, window_fn=torch.hann_window,
                               f_min=f_min, f_max=f_max, wkwargs={'device':y.device}, mel_scale='slaney').to(y.device)
    return transform(y)


def imel_approx(mel_spectrogram, sr, n_fft, n_mels, win_length,
                hop_length, f_min=0., f_max=None, noisy_phase=None):
    """
    inverse mel spectrogram to wav, specifically, mel -> power spec -> wav
    :return: (B, T)
    """
    trans_mel = InverseMelScale(sample_rate=sr, n_stft=n_fft // 2 + 1, n_mels=n_mels,
                                f_min=f_min, f_max=f_max).to(mel_spectrogram.device)
    print(mel_spectrogram.requires_grad)
    mel_spectrogram.requires_grad = False
    power_spec = trans_mel(mel_spectrogram)
    griffin_lim = GriffinLim(n_fft=n_fft, n_iter=32, win_length=win_length,
                             hop_length=hop_length, window_fn=torch.hann_window,
                             wkwargs={'device':mel_spectrogram.device}).to(mel_spectrogram.device)
    return griffin_lim(power_spec)

def imel_phase(mel_spectrogram, noisy_phase, sr, n_fft, n_mels, win_length,
                hop_length, f_min=0., f_max=None):
    """
    inverse mel spectrogram using noisy phase to wav, specifically, mel -> power spec -> wav
    :return: (B, T)
    """
    trans_to_power = InverseMelScale(sample_rate=sr, n_stft=n_fft // 2 + 1, n_mels=n_mels,
                                f_min=f_min, f_max=f_max).to(mel_spectrogram.device)
    print(mel_spectrogram.requires_grad)
    mel_spectrogram.requires_grad = False
    power_spec = trans_to_power(mel_spectrogram)
    mag_spec = torch.sqrt(power_spec)
    complex_spec = mag_spec * torch.exp(1j * noisy_phase)
    trans_to_wav = InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window_fn=torch.hann_window, wkwargs={'device':mel_spectrogram.device}).to(mel_spectrogram.device)
    return trans_to_wav(complex_spec)



def istft(features, n_fft, hop_length, win_length, length=None, input_type="complex"):
    """Wrapper of the official torch.istft.

    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")

    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        assert isinstance(features, tuple) or isinstance(features, list), "Features must be a tuple or list."
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        assert torch.is_complex(features), "Features must be complex-valued."
    elif input_type == "mag_phase":
        assert isinstance(features, tuple) or isinstance(features, list), "Features must be a tuple or list."
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported."
        )

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=features.device),
        length=length,
    )


def mag_phase(complex_tensor):
    mag, phase = torch.abs(complex_tensor), torch.angle(complex_tensor)
    return mag, phase


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    # dB = 20 * log10(x / x_ref)
    rms = np.sqrt(np.mean(y**2)) # rms is the root-mean-square of the signal
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps) # compute 1 / x_ref
    y *= scalar # normalize use x_ref
    return y, rms, scalar # return normalized y


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold) # any() return True if any element of the iterable is true. If not, return False.


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples

    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)

        Only support subsample in the last axis.
    """
    assert data_a.shape[-1] == data_b.shape[-1], "Inconsistent dataset size."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        # data_a = data_a[..., start: end]
        return data_a[..., start:end], data_b[..., start:end]
    elif data_a.shape[-1] < sub_sample_length:
        length = data_a.shape[-1]
        pad_size = sub_sample_length - length
        pad_width = [(0, 0)] * (data_a.ndim - 1) + [(0, pad_size)]
        data_a = np.pad(data_a, pad_width=pad_width, mode="constant", constant_values=0)
        data_b = np.pad(data_b, pad_width=pad_width, mode="constant", constant_values=0)
        return data_a, data_b
    else:
        return data_a, data_b


def subsample(
    data, sub_sample_length, start_position: int = -1, return_start_position=False
):
    """Randomly select fixed-length data from.

    Args:
        data: **one-dimensional data**
        sub_sample_length: how long
        start_position: If start index smaller than 0, randomly generate one index
        return_start_position: add start position return or not

    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
    else:
        return data


def overlap_cat(chunk_list, dim=-1):
    """Overlap concatenate (50%) a list of tensors.

    Args:
        dim: which dimension to concatenate
        chunk_list(list): [[B, T], [B, T], ...]

    Returns:
        concatenated tensor of shape [B, T]
    """
    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):
    """Return the percentage of the time the audio signal is above an energy threshold

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + eps)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def batch_shuffle_frequency(tensor, indices=None):
    """Randomly shuffle frequency of a spectrogram and return shuffle indices.

    Args:
        tensor: input tensor with batch dim
        indices:

    Examples:
        input =
            tensor([[[[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.],
                      [4., 4., 4.]]],
                    [[[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.],
                      [4., 4., 4.]]]])

        output =
            tensor([[[[3., 3., 3.],
                      [4., 4., 4.],
                      [2., 2., 2.],
                      [1., 1., 1.]]],
                    [[[3., 3., 3.],
                      [2., 2., 2.],
                      [1., 1., 1.],
                      [4., 4., 4.]]]])

    Shapes:
        tensor: [B, C, F, T]
        out: [B, C, F T]
        indices: [B, C, F, T]

    Returns:
        out: after frequency shuffle
        indices: shuffle matrix
    """
    assert tensor.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = tensor.shape

    if not torch.is_tensor(indices):
        indices = torch.stack(
            [torch.randperm(num_freqs, device=tensor.device) for _ in range(batch_size)],
            dim=0,
        )
        indices = indices[:, None, :, None].repeat(1, num_channels, 1, num_frames)

    out = torch.gather(tensor, dim=2, index=indices)
    return out, indices


def drop_band(x, num_groups=2):
    """Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = x.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return x

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        x = x[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = x.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(
            group_idx, batch_size, num_groups, device=x.device
        )
        freqs_indices = torch.arange(
            group_idx, num_freqs, num_groups, device=x.device
        )

        selected_samples = torch.index_select(
            x, dim=0, index=samples_indices
        )
        selected = torch.index_select(
            selected_samples, dim=2, index=freqs_indices
        )  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


class ChannelWiseLayerNorm(nn.LayerNorm):
    """Channel wise layer normalization"""

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x
