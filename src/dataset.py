import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class EventDetectionDataset(Dataset):
    def __init__(self, npy_paths, labels=None, window_size=125, is_train=True, stats=None):
        self.npy_paths = npy_paths
        self.labels = labels
        self.window_size = window_size
        self.is_train = is_train
        self.rng = np.random.default_rng()
        # stats = (mean, std)
        self.stats = stats

    def __len__(self):
        return len(self.npy_paths)

    def _prepare_sample(self, feature):
        # Padding (silence)
        if feature.shape[-1] < self.window_size:
            pad_width = self.window_size - feature.shape[-1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode="constant")

        # Crop
        if feature.shape[-1] > self.window_size:
            max_shift = feature.shape[-1] - self.window_size
            start = self.rng.integers(0, max_shift + 1) if self.is_train else max_shift // 2
            feature = feature[:, start : start + self.window_size]

        return feature

    def __getitem__(self, idx):
        feature = np.load(self.npy_paths[idx])

        if self.stats:
            mean, std = self.stats
            feature = (feature - mean) / (std + 1e-7)

        feature = self._prepare_sample(feature)

        x_tensor = torch.from_numpy(feature).float().unsqueeze(0)

        if self.labels is not None:
            return x_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        return x_tensor


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=32,
        window_size=125,
        num_workers: int = 4,
        stats=None,
    ):
        super().__init__()
        self.data_train = (x_train, y_train)
        self.data_val = (x_val, y_val)
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_workers = num_workers
        self.stats = stats

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = EventDetectionDataset(
                *self.data_train,
                window_size=self.window_size,
                is_train=True,
                stats=self.stats,
            )
            self.val_ds = EventDetectionDataset(
                *self.data_val, window_size=self.window_size, is_train=False, stats=self.stats
            )

        if stage == "test":
            self.test_ds = EventDetectionDataset(
                *self.data_test, window_size=self.window_size, is_train=False, stats=self.stats
            )

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
