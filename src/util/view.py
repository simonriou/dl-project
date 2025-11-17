import os
import random
import matplotlib.pyplot as plt
import soundfile as fs
import torch
import numpy as np

from constants import *

def plot_feature(dir, fname=None, compare_speech=False):
    if fname is None:
        fname = random.choice(os.listdir(dir))
    file_path = os.path.join(dir, fname)

    if compare_speech:
        file_id = fname.split('_')[0]
        speech_fname = f"{file_id}.flac"
        speech_path = os.path.join('data/train/speech/', speech_fname)
        # Load and plot the speech signal
        speech_signal, sr = fs.read(speech_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Loading comparison speech | Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}")
        
        if speech_signal.ndim > 1:
            speech_signal = speech_signal.mean(axis=1)
        
        speech_signal_t = torch.tensor(speech_signal, dtype=torch.float32)

        S = torch.stft(
            speech_signal_t,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH),
            return_complex=True
        )

        S_power = torch.abs(S) ** 2
        log_S = torch.log(S_power + EPSILON)

        spectro = log_S.numpy()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(spectro, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Speech Spectrogram: {speech_fname}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        feature_spectro = np.load(file_path)
        plt.imshow(feature_spectro, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Feature Spectrogram: {fname}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        # plt.tight_layout()
        plt.show()

    else:
        # Plot the feature (spectrogram)
        spectro = np.load(file_path)
        plt.figure(figsize=(10, 4))
        plt.imshow(spectro, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {fname}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        # plt.tight_layout()
        plt.show()