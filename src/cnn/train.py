import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from cnn.cnnMask import build_cnn_mask_model
from cnn.data import SpectrogramDataset
from tqdm import tqdm

from constants import *

def train_cnn_model(log_dir, features_dir, labels_dir):
    os.makedirs(f"{log_dir}/models/", exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints/", exist_ok=True)
    os.makedirs(f"{log_dir}/history/", exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Build model
    model = build_cnn_mask_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Datasets & loaders
    train_ds = SpectrogramDataset(
        features_dir=features_dir,
        labels_dir=labels_dir,
        validation_split=VAL_SPLIT,
        subset="training"
    )
    val_ds = SpectrogramDataset(
        features_dir=features_dir,
        labels_dir=labels_dir,
        validation_split=VAL_SPLIT,
        subset="validation"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Training loop
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        # Training
        model.train()
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (X, y) in enumerate(train_loader_tqdm):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_loader_tqdm.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for X_val, y_val in val_loader_tqdm:
                X_val, y_val = X_val.to(device), y_val.to(device)

                y_pred = model(X_val)
                loss = criterion(y_pred, y_val)
                val_loss += loss.item() * X_val.size(0)

                val_loader_tqdm.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        val_loss /= len(val_ds)

        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        if epoch > 0:
            train_diff = train_loss_history[-1] - train_loss
            val_diff = val_loss_history[-1] - val_loss
            print(f"Train loss improvement: {train_diff:.6f} | Val loss improvement: {val_diff:.6f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Save checkpoint
        checkpoint_path = f"{log_dir}/checkpoints/cnn_epoch_{epoch+1:02d}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), f"{log_dir}/models/cnn_final.pt")
    
    # Save training history
    np.save(f"{log_dir}/history/train_loss_history.npy", np.array(train_loss_history))
    np.save(f"{log_dir}/history/val_loss_history.npy", np.array(val_loss_history))

    print("\n" + "="*60)
    print("Training complete.")
    print(f"Final train loss: {train_loss_history[-1]:.6f}")
    print(f"Final val loss: {val_loss_history[-1]:.6f}")
    print(f"Best val loss: {min(val_loss_history):.6f} (epoch {val_loss_history.index(min(val_loss_history))+1})")
    print("="*60 + "\n")