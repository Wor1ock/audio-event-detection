import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ASTAudioDataset
from src.model import ASTClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description="AST Inference on raw audio.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--labels-pickle", type=Path, default=Path("data/labels.pickle"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.open())

    with args.labels_pickle.open("rb") as f:
        label_to_id = pickle.load(f)

    id_to_label = {idx: label for label, idx in label_to_id.items()}

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = sorted(Path("models/ast_run/").glob("*.ckpt"))[-1]

    model = ASTClassificationModel.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path), num_classes=cfg["model"]["num_classes"]
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dir = Path(cfg["data"]["test_audio_dir"])
    test_files = sorted(test_dir.glob("*.wav"))
    test_paths = [str(p) for p in test_files]
    fnames = [p.name for p in test_files]

    dm_config = {
        "sample_rate": cfg["audio"]["sample_rate"],
        "use_specaug": False,
        "batch_size": cfg["training"]["batch_size"] * 2,
    }

    dataset = ASTAudioDataset(audio_paths=test_paths, labels=None, config=dm_config, is_train=False)

    loader = DataLoader(dataset, batch_size=dm_config["batch_size"], shuffle=False, num_workers=4)

    results = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(loader, desc="AST Predicting")):
            # batch: [B, 1024, 128]
            x = batch.to(device)
            logits = model(x)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            start_idx = i * dm_config["batch_size"]
            for j, pred_id in enumerate(preds):
                results.append({"fname": fnames[start_idx + j], "label": id_to_label[int(pred_id)]})

    out_df = pd.DataFrame(results)
    out_path = Path(cfg["data"]["submission_csv"])
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
