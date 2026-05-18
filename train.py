import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.dataset import AudioDataModule
from src.features import FeatureAugmentation
from src.model import ASTAudioClassifier, AudioTrainingSystem
from src.utils import set_seed


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    train_transform = None
    if cfg.data.use_augmentation:
        train_transform = FeatureAugmentation(**cfg.augmentation)

    dm = AudioDataModule(
        train_transform=train_transform,
        **cfg.data,
    )

    abs_model_path = hydra.utils.to_absolute_path(str(cfg.model.model_path))
    net = ASTAudioClassifier(
        model_path=abs_model_path,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )

    system = AudioTrainingSystem(
        model=net,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
    )

    checkpoint_dir = hydra.utils.to_absolute_path(cfg.trainer.checkpoint_dir)
    log_dir = hydra.utils.to_absolute_path(cfg.trainer.log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=cfg.trainer.patience,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.trainer.gradient_accumulation_steps,
        logger=[
            CSVLogger(log_dir, name="ast_audio"),
            TensorBoardLogger(log_dir, name="ast_tb"),
        ],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(system, datamodule=dm)


if __name__ == "__main__":
    train()
