import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EventDetectionDataset
from src.model import AudioClassificationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on preprocessed test features.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--test-pickle", type=Path, default=None, help="Path to preprocessed test pickle")
    parser.add_argument("--labels-pickle", type=Path, default=None, help="Path to label_to_id pickle")
    parser.add_argument("--stats-pickle", type=Path, default=None, help="Path to train stats")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to Lightning checkpoint")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    model_dir = Path("models")
    candidates = sorted(model_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .ckpt files found in models/. Pass --checkpoint explicitly.")
    return candidates[0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    test_pickle = args.test_pickle or Path(cfg["data"]["test_pickle"])
    labels_pickle = args.labels_pickle or Path(cfg["data"]["labels_pickle"])
    stats_pickle = args.stats_pickle or Path(cfg["data"]["stats_pickle"])
    checkpoint = resolve_checkpoint(args.checkpoint)

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["training"].get("num_workers", 0))
    output_csv = args.output_csv or Path(cfg["data"].get("submission_csv", "data/submission.csv"))

    with test_pickle.open("rb") as f:
        test_rows = pickle.load(f)

    with labels_pickle.open("rb") as f:
        label_to_id = pickle.load(f)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    if not stats_pickle.exists():
        raise FileNotFoundError(f"Stats not found at {stats_pickle}. Did you save them in train.py?")
    with stats_pickle.open("rb") as f:
        train_stats = pickle.load(f)

    window_size = int((cfg["audio"]["duration_sec"] * cfg["audio"]["sample_rate"]) / cfg["audio"]["hop_length"])
    n_mels = cfg["audio"]["n_mels"]
    num_classes = cfg["model"]["num_classes"]

    model = AudioClassificationModel.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        num_classes=num_classes,
        input_shape=(1, n_mels, window_size),
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_paths = [row["npy_path"] for row in test_rows]
    test_fnames = [row["fname"] for row in test_rows]

    dataset = EventDetectionDataset(
        npy_paths=test_paths, labels=None, window_size=window_size, is_train=False, stats=train_stats
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0 if num_workers else False,
    )

    results = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(loader, desc="Predicting")):
            x = batch.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            start_idx = i * batch_size
            for j, pred_id in enumerate(preds):
                fname = test_fnames[start_idx + j]
                results.append({"fname": fname, "label": id_to_label[int(pred_id)]})

    out_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
