import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch

from constants import *

def extract_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir), desc="Extracting features"):
        if not fname.lower().endswith(('.flac', '.wav')):
            continue

        path = os.path.join(input_dir, fname)

        # Load the audio
        y, sr = sf.read(path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Loading feature audio | Expected sample rate {SAMPLE_RATE}, but got {sr}")
        
        # Normalize audio
        y = y / (np.max(np.abs(y)) + EPSILON)

        # Convert to mono if needed
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Convert to a 1D tensor
        y_t = torch.tensor(y, dtype=torch.float32)

        # STFT
        S = torch.stft(
            y_t,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH),
            return_complex=True
        )

        # Power spectrogram
        S_power = torch.abs(S) ** 2

        # Log-spectrogram
        log_S = torch.log(S_power + EPSILON)

        # Save as .npy
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, base + '.npy')
        np.save(out_path, log_S.numpy())

    print("Feature extraction completed")
