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
    set_seed(cfg.training.random_state)

    abs_model_path = hydra.utils.to_absolute_path(str(cfg.model.model_path))

    train_transform = FeatureAugmentation(
        num_mel_bins=cfg.audio.num_mel_bins,
        max_proportion=cfg.augmentation.max_proportion,
        time_mask_param=cfg.augmentation.time_mask_param,
    )

    dm = AudioDataModule(
        train_pickle_path=hydra.utils.to_absolute_path(cfg.data.train_meta_pickle),
        test_pickle_path=hydra.utils.to_absolute_path(cfg.data.test_meta_pickle),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        test_size=cfg.data.test_size,
        random_state=cfg.training.random_state,
        train_transform=train_transform,
    )

    net = ASTAudioClassifier(
        model_path=abs_model_path,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.get("dropout", 0.0),
    )

    model_module = AudioTrainingSystem(
        model=net,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
    )

    checkpoint_dir = hydra.utils.to_absolute_path(cfg.training.checkpoint_dir)
    log_dir = hydra.utils.to_absolute_path(cfg.training.log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=cfg.training.patience,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        logger=[
            CSVLogger(log_dir, name="ast_audio"),
            TensorBoardLogger(log_dir, name="ast_tb"),
        ],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model_module, datamodule=dm)


if __name__ == "__main__":
    train()
