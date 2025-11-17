import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, features_dir, labels_dir, validation_split=0.0, subset=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        # List all feature files
        self.files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])

        # Validation split
        if validation_split > 0.0:
            if subset not in {"training", "validation"}:
                raise ValueError("subset must be 'training' or 'validation'")
            split_idx = int(len(self.files) * (1 - validation_split))
            if subset == "training":
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        feat_path = os.path.join(self.features_dir, fname)
        label_fname = fname.split('_')[0] + '_ibm.npy'
        label_path = os.path.join(self.labels_dir, label_fname)

        feat = np.load(feat_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        # Add channel dimension (C, F, T)
        feat = np.expand_dims(feat, axis=0)
        label = np.expand_dims(label, axis=0)

        return torch.from_numpy(feat), torch.from_numpy(label)