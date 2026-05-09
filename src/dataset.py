from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import AudioFeatureExtractor


def apply_mixin(waveform: np.ndarray, noise_paths: list[Path], snr_db: int = 15) -> np.ndarray:
    if not noise_paths:
        return waveform

    rng = np.random.default_rng()
    noise_path = rng.choice(noise_paths)

    noise, _ = librosa.load(noise_path, sr=16000)

    if len(noise) > len(waveform):
        noise = noise[: len(waveform)]
    else:
        pad_len = len(waveform) - len(noise)
        noise = np.pad(noise, (0, pad_len), mode="constant")

    p_wav = np.mean(waveform**2)
    p_noise = np.mean(noise**2)

    if p_noise == 0:
        return waveform

    k = np.sqrt(p_wav / (10 ** (snr_db / 10) * p_noise))
    mixed = waveform + k * noise

    return mixed.astype(np.float32)


class ASTAudioDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[str],
        labels: list[int] | None = None,
        config: dict = None,
        is_train: bool = True,
        noise_paths: list[Path] = None,
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.is_train = is_train
        self.noise_paths = noise_paths
        self.config = config or {}
        self.rng = np.random.default_rng()

        self.extractor = AudioFeatureExtractor(
            sample_rate=self.config.get("sample_rate", 16000),
            need_augment=is_train and self.config.get("use_specaug", False),
            max_proportion=self.config.get("max_proportion", 0.3),
        )

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        wav_path = self.audio_paths[idx]
        waveform, _ = librosa.load(wav_path, sr=16000)

        if self.is_train and self.noise_paths and self.rng.random() < self.config.get("mixin_prob", 0.0):
            waveform = apply_mixin(waveform, self.noise_paths, self.config.get("snr_db", 15))

        with torch.no_grad():
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            feature = self.extractor(waveform_t)

        feature = feature.squeeze(0)

        if self.labels is not None:
            return feature, torch.tensor(self.labels[idx], dtype=torch.long)
        return feature


class AudioDataModule(L.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict, noise_paths: list[Path] = None):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config
        self.noise_paths = noise_paths

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = ASTAudioDataset(
                audio_paths=self.train_df["audio_path"].tolist(),
                labels=self.train_df["label_id"].tolist(),
                config=self.config,
                is_train=True,
                noise_paths=self.noise_paths,
            )
            self.val_ds = ASTAudioDataset(
                audio_paths=self.val_df["audio_path"].tolist(),
                labels=self.val_df["label_id"].tolist(),
                config=self.config,
                is_train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"]
        )
