import torch
import numpy as np
from audio.load import load_audio

from constants import *

def audio_to_spectrogram(filepath):
    y, sr = load_audio(filepath)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Loading audio | Expected sample rate {SAMPLE_RATE}, but got {sr}")
    
    if y.ndim > 1:
            y = y.mean(axis=1)

    # Normalize audio
    y = y / (np.max(np.abs(y)) + EPSILON)

    y_t = torch.tensor(y, dtype=torch.float32)

    S = torch.stft(
        y_t,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        return_complex=True
    )
    mag, phase = torch.abs(S), torch.angle(S)
    mag = mag[np.newaxis, np.newaxis, :, :] # (1, C, F, T)
    return mag, phase, y.shape[0]

def spectrogram_to_audio(mag, phase):
    S_reconstructed = mag.squeeze(0).squeeze(0) * torch.exp(1j * phase)
    y_reconstructed = torch.istft(
        S_reconstructed,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        length=None
    )
    return y_reconstructed.numpy()