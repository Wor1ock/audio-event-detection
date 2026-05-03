import pickle
from pathlib import Path

import pytorch_lightning as L
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import StratifiedShuffleSplit

from src.dataset import AudioDataModule
from src.model import AudioClassificationModel
from src.utils import compute_norm_stats, plot_metrics_from_log, set_seed


def load_config(config_path: str = "config.yaml") -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def train() -> None:
    cfg = load_config()
    set_seed(cfg["training"]["random_state"])

    with Path(cfg["data"]["train_pickle"]).open("rb") as f:
        train_data = pickle.load(f)

    all_npy_paths = [row["npy_path"] for row in train_data]
    all_labels = [row["label_id"] for row in train_data]

    window_size = int((cfg["audio"]["duration_sec"] * cfg["audio"]["sample_rate"]) / cfg["audio"]["hop_length"])
    n_mels = cfg["audio"]["n_mels"]
    num_classes = cfg["model"]["num_classes"]
    num_workers = cfg["training"].get("num_workers", 0)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg["data"]["test_size"], random_state=cfg["training"]["random_state"]
    )

    train_idx, val_idx = next(sss.split(all_npy_paths, all_labels))

    x_tr = [all_npy_paths[i] for i in train_idx]
    y_tr = [all_labels[i] for i in train_idx]
    x_val = [all_npy_paths[i] for i in val_idx]
    y_val = [all_labels[i] for i in val_idx]

    train_stats = compute_norm_stats(x_tr)
    with Path(cfg["data"]["stats_pickle"]).open("wb") as f:
        pickle.dump(train_stats, f)

    dm = AudioDataModule(
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_val,
        y_val=y_val,
        batch_size=cfg["training"]["batch_size"],
        window_size=window_size,
        num_workers=num_workers,
        stats=train_stats,
    )

    # input_shape: (каналы=1, n_mels, window_size)
    input_shape = (1, n_mels, window_size)
    model = AudioClassificationModel(
        num_classes=num_classes,
        input_shape=input_shape,
        lr=cfg["model"]["lr"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg["training"]["patience"], mode="min")
    csv_logger = CSVLogger("logs", name="audio_rnd")
    tb_logger = TensorBoardLogger("logs", name="tb_logs")

    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic="warn",
    )

    trainer.fit(model, datamodule=dm)
    plot_metrics_from_log(f"{csv_logger.log_dir}/metrics.csv")


if __name__ == "__main__":
    train()
