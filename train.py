from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import StratifiedShuffleSplit

from src.dataset import AudioDataModule
from src.model import ASTClassificationModel
from src.utils import set_seed


def load_config(config_path: str = "config.yaml") -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    set_seed(cfg["training"]["random_state"])

    df = pd.read_csv(cfg["data"]["train_csv"])
    audio_dir = Path(cfg["data"]["train_audio_dir"])

    df["audio_path"] = df["fname"].apply(lambda x: str(audio_dir / x))

    labels = sorted(df["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label_to_id)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg["data"]["test_size"], random_state=cfg["training"]["random_state"]
    )
    train_idx, val_idx = next(sss.split(df["audio_path"], df["label_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    noise_dir = Path(cfg["data"]["noise_dir"])
    noise_paths = list(noise_dir.glob("*.wav"))

    dm_config = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": cfg["training"]["num_workers"],
        "sample_rate": cfg["audio"]["sample_rate"],
        "use_specaug": cfg["augmentation"]["use_specaug"],
        "max_proportion": cfg["augmentation"]["max_proportion"],
        "mixin_prob": cfg["augmentation"]["mixin_prob"],
        "snr_db": cfg["augmentation"]["snr_db"],
    }

    dm = AudioDataModule(train_df=train_df, val_df=val_df, config=dm_config, noise_paths=noise_paths)

    model = ASTClassificationModel(
        num_classes=cfg["model"]["num_classes"], lr=cfg["model"]["lr"], weight_decay=cfg["model"]["weight_decay"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/ast_run/",
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg["training"]["patience"], mode="min")

    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg["training"]["gradient_accumulation_steps"],
        logger=[CSVLogger("logs", name="ast_audio"), TensorBoardLogger("logs", name="ast_tb")],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
