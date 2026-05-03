import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from tqdm import tqdm


def set_seed(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    L.seed_everything(seed, workers=True)

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_norm_stats(npy_paths):
    means = []
    stds = []
    for path in tqdm(npy_paths, desc="Computing stats"):
        data = np.load(path)
        means.append(data.mean())
        stds.append(data.std())
    return np.mean(means), np.mean(stds)


def plot_metrics_from_log(log_path: str):
    if not Path(log_path).exists():
        print(f"Лог-файл {log_path} не найден.")
        return

    metrics = pd.read_csv(log_path)

    metrics_epoch = metrics.groupby("epoch").mean().reset_index()
    epochs = metrics_epoch["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    train_loss = metrics_epoch.get("train_loss")
    val_loss = metrics_epoch.get("val_loss")

    if train_loss is not None:
        ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
    if val_loss is not None:
        ax1.plot(epochs, val_loss, "r-", label="Valid Loss", linewidth=2)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    train_acc = metrics_epoch.get("train_acc")
    val_acc = metrics_epoch.get("val_acc")

    if train_acc is not None:
        ax2.plot(epochs, train_acc, "b-", label="Training Accuracy", linewidth=2)
    if val_acc is not None:
        ax2.plot(epochs, val_acc, "r-", label="Valid Accuracy", linewidth=2)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
