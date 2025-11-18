import torch
from constants import *
import math

def hz_to_mel(hz):
    return 2595 * math.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def create_mel_filterbank(fmin=0.0, fmax=8000.0):
    if fmax is None:
        fmax = SAMPLE_RATE / 2

    # Compute mel points
    mal_points = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), NMELS + 2)
    hz_points = mel_to_hz(mal_points)

    bin_freqs = torch.floor((NFFT + 1) * hz_points / SAMPLE_RATE).long()

    # Filter bank
    fb = torch.zeros(NMELS, NFFT // 2 + 1)

    for m in range(1, NMELS + 1):
        f_m_minus, f_m, f_m_plus = bin_freqs[m - 1], bin_freqs[m], bin_freqs[m + 1]
        if f_m > f_m_minus:
            fb[m - 1, f_m_minus:f_m] = torch.linspace(0, 1, f_m - f_m_minus)
        if f_m_plus > f_m:
            fb[m - 1, f_m:f_m_plus] = torch.linspace(1, 0, f_m_plus - f_m)
    return fb