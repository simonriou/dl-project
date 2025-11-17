import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random

from constants import *

def add_noise(speech_dir, noise_type, snr_db, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    noise_dir = f'data/noise/{noise_type}'
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

    target_len = int(SAMPLE_RATE * DURATION)

    for fname in tqdm(os.listdir(speech_dir)):
        if not fname.lower().endswith('flac'):
            continue

        speech_path = os.path.join(speech_dir, fname)

        # Load the speech file
        speech, sr = sf.read(speech_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Loading speech | Expected sample rate {SAMPLE_RATE}, but got {sr}")
        
        L = len(speech)

        # ------- 1: Adjust duration -------
        if L > target_len:
            # Crop randomly (to avoid always taking the start)
            start = random.randint(0, L - target_len)
            speech_trunc = speech[start:start + target_len]
        else:
            # Pad with EPISLONS
            pad_width = target_len - L
            speech_trunc = np.pad(speech, (0, pad_width), mode='constant', constant_values=EPSILON)

        # Replace the original speech signal by the adjusted one
        # Temp path (Macos/Linux only)
        tmp_path = speech_path + ".tmp"
        sf.write(tmp_path, speech_trunc, SAMPLE_RATE, format='FLAC')
        os.replace(tmp_path, speech_path)

        # ------- 2: Select random file & segment -------
        noise_file = random.choice(noise_files)
        noise, sr = sf.read(os.path.join(noise_dir, noise_file))
        if sr != SAMPLE_RATE:
            raise ValueError(f"Loading noise | Expected sample rate {SAMPLE_RATE}, but got {sr}")
        
        N = len(noise)
        start_idx = random.randint(0, N - target_len)
        noise_seg = noise[start_idx:start_idx + target_len]

        # ------- 3: Add noise at fixed SNR -------
        # Normalize speech & noise
        speech_trunc = speech_trunc / np.max(np.abs(speech_trunc) + EPSILON)
        noise_seg = noise_seg / np.max(np.abs(noise_seg) + EPSILON)

        speech_power = np.mean(speech_trunc ** 2)
        noise_power = np.mean(noise_seg ** 2)

        alpha = np.sqrt(speech_power / ((noise_power + EPSILON) * (10 ** (snr_db / 10))))

        noisy = speech_trunc + alpha * noise_seg

        # Normalize to avoid clipping
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val

        # ------- 4: Save noisy file -------
        # Encode noise start index, noise file, noise type in file name
        base = os.path.splitext(fname)[0] # "xx-xxxxxx-xxxx"
        out_name = f"{base}_noisy_{start_idx}_{noise_type}_{noise_file.replace('.wav', '')}.flac"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, noisy, SAMPLE_RATE, format='FLAC')

    print("All files processed.")