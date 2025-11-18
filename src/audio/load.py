import av
import numpy as np

def load_audio(path):
    container = av.open(path)
    stream = container.streams.audio[0]
    sr = stream.rate

    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        frames.append(arr)

    if not frames:
        raise ValueError(f"Could not decode audio file: {path}")
    
    audio = np.concatenate(frames, axis=1).astype(np.float32)

    if audio.shape[0] > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio[0]

    return audio, sr