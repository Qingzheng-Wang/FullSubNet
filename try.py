#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        try.py
# Author:           Qingzheng WANG
# Time:             2023/11/21 14:37
# Description:      show the phase spectrogram
# Function List:
# ===================================================

import scipy.io.wavfile as wav
from torch import stft, istft
import matplotlib.pyplot as plt
from torch import hann_window
import torch

if __name__ == '__main__':
    import scipy.io.wavfile as wav
    from torch import stft
    import matplotlib.pyplot as plt
    from torch import hann_window
    import torch

    if __name__ == '__main__':
        # 读取wav文件
        fs, data = wav.read(
            "/mnt/inspurfs/home/wangqingzheng/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/clean/clean_fileid_96.wav")
        data = torch.from_numpy(data).float()

        # 计算STFT
        c = stft(data, n_fft=1024, hop_length=256, win_length=1024, window=hann_window(1024), return_complex=True)
        mag = torch.abs(c)
        # normalize mag
        mag = (mag - mag.min()) / (mag.max() - mag.min())

        # 显示谱图和颜色条
        plt.figure(figsize=(10, 6))
        im = plt.imshow(mag.numpy(), aspect='auto', cmap='viridis', origin='lower')

        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('Magnitude')

        plt.title("Magnitude Spectrogram")
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()
