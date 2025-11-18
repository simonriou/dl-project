import os
import numpy as np
import torch
from tqdm import tqdm

from audio.load import load_audio
from constants import *

def compute_IBM(signal, noise):
    signal = signal / (np.max(np.abs(signal)) + EPSILON)
    noise = noise / (np.max(np.abs(noise)) + EPSILON)

    signal_t = torch.tensor(signal, dtype=torch.float32)
    noise_t = torch.tensor(noise, dtype=torch.float32)

    S = torch.stft(
        signal_t,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        return_complex=True
    )

    N = torch.stft(
        noise_t,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        return_complex=True
    )

    S_power = torch.abs(S) ** 2
    N_power = torch.abs(N) ** 2

    IBM = (S_power >= N_power).float()
    return IBM.numpy()

def extract_labels(speech_dir, noisy_dir, noise_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    noisy_files = os.listdir(noisy_dir)

    for fname in tqdm(os.listdir(speech_dir), desc="Extracting labels"):
        if not fname.lower().endswith('.flac'):
            continue

        speech_id = os.path.splitext(os.path.basename(fname))[0]

        # Find the corresponding noisy file (the one that contains speech_id)
        noisy_file = None
        for f in noisy_files:
            if speech_id in f:
                noisy_file = f
                break

        if noisy_file is None:
            raise FileNotFoundError(f"No corresponding noisy file found for {fname}")
        
        # Format is {speech_id}_noisy_{start_idx}_{noise_type}_{noise_file}.flac (no _ in noise_file)
        start_idx = int(noisy_file.split('_')[2])
        noise_type = noisy_file.split('_')[3]
        noise_file_name = noisy_file.split('_')[4].replace('.flac', '.wav')
        fpath = f"{noise_dir}/{noise_type}/{noise_file_name}"

        # Load speech and noise signals
        speech_signal, sr = load_audio(f"{speech_dir}/{fname}")
        if sr != SAMPLE_RATE:
            raise ValueError(f"Importing speech signal | Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}")
        noise_signal, sr = load_audio(fpath)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Importing noise signal | Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}")
        
        noise_segment = noise_signal[start_idx:start_idx + len(speech_signal)]

        # Compute IBM
        ibm = compute_IBM(speech_signal, noise_segment)

        # Save as .npy to output_dir
        ibm_output_path = os.path.join(output_dir, f"{speech_id}_ibm.npy")
        np.save(ibm_output_path, ibm)

    print("Label extraction completed.")