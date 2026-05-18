import os

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as L
import torch


def set_seed(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    L.seed_everything(seed, workers=True)

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_metrics_from_log(log_path: str):
    metrics = pd.read_csv(log_path)

    metrics_epoch = metrics.groupby("epoch").mean().reset_index()
    epochs = metrics_epoch["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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

    train_f1 = metrics_epoch.get("train_f1")
    val_f1 = metrics_epoch.get("val_f1")

    if train_f1 is not None:
        ax2.plot(epochs, train_f1, "b-", label="Training F1", linewidth=2)
    if val_f1 is not None:
        ax2.plot(epochs, val_f1, "r-", label="Valid F1", linewidth=2)
    ax2.set_title("F1 Score")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
