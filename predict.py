import pickle
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.dataset import AudioDataModule
from src.model import ASTAudioClassifier, AudioTrainingSystem


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    with Path(hydra.utils.to_absolute_path(cfg.data.labels_pickle)).open("rb") as f:
        labels_data = pickle.load(f)
    label2id = labels_data["label2id"]
    id2label = labels_data["id2label"]
    num_classes = len(label2id)

    checkpoint_path = cfg.prediction.get("checkpoint_path") if hasattr(cfg, "prediction") else None
    if checkpoint_path is None:
        checkpoint_dir = Path(hydra.utils.to_absolute_path(cfg.training.checkpoint_dir))
        checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

    abs_model_path = hydra.utils.to_absolute_path(str(cfg.model.model_path))
    net = ASTAudioClassifier(
        model_path=abs_model_path,
        num_classes=num_classes,
        dropout=0.0,
    )

    system = AudioTrainingSystem.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=net,
        num_classes=num_classes,
    )
    system.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = system.to(device)

    dm = AudioDataModule(
        train_pickle_path=hydra.utils.to_absolute_path(cfg.data.train_meta_pickle),
        test_pickle_path=hydra.utils.to_absolute_path(cfg.data.test_meta_pickle),
        batch_size=cfg.training.batch_size * 2,
        num_workers=cfg.training.num_workers,
        test_size=cfg.data.test_size,
        random_state=cfg.training.random_state,
        train_transform=None,
    )

    dm.setup(stage="test")
    loader = dm.test_dataloader()

    results = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            logits = system(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            results.extend(preds.tolist())

    submission_path = Path(hydra.utils.to_absolute_path(cfg.data.submission_csv))
    out_df = pd.read_csv(submission_path)

    out_df["label"] = [id2label[pred_id] for pred_id in results[: len(out_df)]]
    out_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
