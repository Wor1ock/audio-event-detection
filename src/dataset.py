import pickle
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset


class ASTAudioDataset(Dataset):
    def __init__(self, samples: list[dict], transform: Callable | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature = torch.load(sample["pt_path"], weights_only=True)

        if self.transform is not None:
            feature = self.transform(feature)

        if sample.get("label_id") is not None:
            return feature, torch.tensor(sample["label_id"], dtype=torch.long)
        return feature


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_pickle_path: str,
        test_pickle_path: str,
        batch_size: int,
        num_workers: int,
        test_size: float,
        seed: int,
        train_transform: Callable | None = None,
    ):
        super().__init__()
        self.train_pickle_path = Path(train_pickle_path)
        self.test_pickle_path = Path(test_pickle_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.seed = seed
        self.train_transform = train_transform

        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            with self.train_pickle_path.open("rb") as f:
                train_df = pd.DataFrame(pickle.load(f))

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
            train_idx, val_idx = next(sss.split(train_df["pt_path"], train_df["label_id"]))

            self.train_samples = train_df.iloc[train_idx].to_dict(orient="records")
            self.val_samples = train_df.iloc[val_idx].to_dict(orient="records")

            self.train_ds = ASTAudioDataset(self.train_samples, transform=self.train_transform)
            self.val_ds = ASTAudioDataset(self.val_samples, transform=None)

        if stage == "test" or stage is None:
            with self.test_pickle_path.open("rb") as f:
                test_df = pickle.load(f)

            self.test_samples = test_df
            self.test_ds = ASTAudioDataset(self.test_samples, transform=None)

    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_ds, shuffle=False)
