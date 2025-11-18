import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def analyse_training_run(log_dir, display_checkpoint: Optional[int] = None):
    print(f"\n=== Analysing training logs in {log_dir} ===\n")

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    history_dir = os.path.join(log_dir, "history")
    models_dir = os.path.join(log_dir, "models")

    if os.path.isdir(ckpt_dir):
        checkpoints = sorted(
            f for f in os.listdir(ckpt_dir) if f.endswith('.pt')
        )
        print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}:")
        for ckpt in checkpoints:
            print(f" - {ckpt}")
    else:
        print(f"\nNo checkpoints directory found at {ckpt_dir}")
        checkpoints = []

    if os.path.isdir(history_dir):
        train_path = os.path.join(history_dir, "train_loss_history.npy")
        val_path = os.path.join(history_dir, "val_loss_history.npy")

        train_loss = (
            np.load(train_path).tolist()
            if os.path.isfile(train_path)
            else []
        )
        val_loss = (
            np.load(val_path).tolist()
            if os.path.isfile(val_path)
            else []
        )

        print(f"\nLoaded {len(train_loss)} training and {len(val_loss)} validation loss values.")

    else:
        print("\nHistory directory not found; cannot load loss curves.")
        train_loss, val_loss = [], []

    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Training Loss")
        if len(val_loss) == len(train_loss):
            plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    final_model_path = os.path.join(models_dir, "cnn_final.pt")
    if os.path.isfile(final_model_path):
        print("\nFinal model found: cnn_final.pt")
        try:
            model_obj = torch.load(final_model_path, map_location='cpu')
            if isinstance(model_obj, dict):
                print(f"Final model is a state_dict ({len(model_obj)} tensors).")
            else:
                print("Final model is a full nn.Module object.")
        except Exception as e:
            print(f"Could not load final model: {e}")
    else:
        print("\nNo final model file found.")

    if display_checkpoint is not None:
        ck_name = f"cnn_epoch_{display_checkpoint:02d}.pt"
        ck_path = os.path.join(ckpt_dir, ck_name)

        print(f"\n=== Inspecting checkpoint epoch {display_checkpoint} ===")
        if os.path.isfile(ck_path):
            try:
                ckpt = torch.load(ck_path, map_location="cpu")
                print(f"Checkpoint content type: {type(ckpt)}")

                if isinstance(ckpt, dict):
                    print(f"Keys: {list(ckpt.keys())}")

                    # If it's a state_dict
                    if "model_state_dict" in ckpt:
                        sd = ckpt["model_state_dict"]
                        total_params = sum(p.numel() for p in sd.values())
                        print(f"Total parameters: {total_params:,}")

            except Exception as e:
                print(f"Could not load checkpoint: {e}")
        else:
            print(f"Checkpoint not found: {ck_name}")