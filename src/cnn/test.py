import os
import torch
import numpy as np
import soundfile as sf
from audio.spectro import *
from cnn.cnnMask import build_cnn_mask_model
from tqdm import tqdm


def test_cnn_model(model_path, test_samples_path, output_dir, mode='ibm', sample_test=False):
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = build_cnn_mask_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_files = [f for f in os.listdir(test_samples_path) if f.endswith(".flac")]
    if sample_test:
        test_files = test_files[:max(1, len(test_files) // 10)]

    for filename in tqdm(test_files, desc="Processing audio files"):
        path = os.path.join(test_samples_path, filename)
        mag, phase, orig_len = audio_to_spectrogram(path)

        if mode.lower() not in ['ibm', 'irm', 'spectro']:
            raise ValueError("Mode must be either 'ibm', 'irm' or 'spectro'")
        
        if mode.lower() in ['ibm', 'irm']:
            # Predict mask
            with torch.no_grad():
                mag_tensor = mag.float().to(device)
                predicted_mask = model(mag_tensor)[0, 0].cpu().numpy()

            # Apply mask
            enhanced_mag = np.abs(mag[0, 0]) * predicted_mask

            # Reconstruct waveform
            enhanced = spectrogram_to_audio(enhanced_mag, phase)
            enhanced = enhanced[:orig_len]

        elif mode.lower() == 'spectro':
            # Predict spectrogram
            with torch.no_grad():
                mag_tensor = mag.float().to(device)
                predicted_spectro = model(mag_tensor)[0, 0].cpu().numpy()

            # Reconstruct waveform
            enhanced = spectrogram_to_audio(predicted_spectro, phase)
            enhanced = enhanced[:orig_len]

        # Save
        out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_enhanced.wav")
        sf.write(out_path, enhanced, SAMPLE_RATE)