import pickle
from pathlib import Path

import hydra
import librosa
import pandas as pd
import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import ASTFeatureExtractor

from src.utils import set_seed


def load_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def trim_silence(
    waveform: torch.Tensor,
    top_db: int,
    frame_length: int,
    hop_length: int,
) -> torch.Tensor:
    waveform_np = waveform.squeeze().cpu().numpy()
    trimmed, _ = librosa.effects.trim(
        waveform_np,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return torch.from_numpy(trimmed).unsqueeze(0)


def build_samples_list(
    df: pd.DataFrame,
    cache_dir: Path,
    label2id: dict | None,
    split: str,
) -> list[dict]:
    samples = []
    for _, row in df.iterrows():
        pt_name = f"{Path(row['fname']).stem}.pt"
        pt_path = cache_dir / split / pt_name

        item = {
            "pt_path": str(pt_path),
            "label_id": label2id[row["label"]] if (label2id is not None and "label" in row) else None,
        }
        samples.append(item)
    return samples


def extract_and_cache_features(
    df: pd.DataFrame,
    samples: list[dict],
    audio_dir: Path,
    extractor: ASTFeatureExtractor,
    target_sample_rate: int,
    trim_config: dict | None,
) -> None:
    for idx, item in enumerate(tqdm(samples, desc="Extracting features")):
        pt_path = Path(item["pt_path"])
        if pt_path.exists():
            continue

        pt_path.parent.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / df.iloc[idx]["fname"]

        waveform = load_audio(str(audio_path), target_sample_rate)

        if trim_config is not None and trim_config.get("enabled", False):
            waveform = trim_silence(
                waveform,
                top_db=trim_config.get("top_db", 20),
                frame_length=trim_config.get("frame_length", 2048),
                hop_length=trim_config.get("hop_length", 512),
            )

        waveform_np = waveform.squeeze(0).numpy()
        inputs = extractor(waveform_np, sampling_rate=target_sample_rate, padding="max_length", return_tensors="pt")

        feature = inputs.input_values.squeeze(0)

        torch.save(feature, pt_path)


def save_labels(df: pd.DataFrame, labels_pickle: str) -> dict:
    unique_labels = sorted(df["label"].dropna().unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    with Path(labels_pickle).open("wb") as f:
        pickle.dump({"label2id": label2id, "id2label": id2label}, f)
    return label2id


def load_labels(labels_pickle: str) -> dict:
    with Path(labels_pickle).open("rb") as f:
        data = pickle.load(f)
    return data["label2id"]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    train_csv = Path(hydra.utils.to_absolute_path(cfg.paths.train_csv))
    train_audio_dir = Path(hydra.utils.to_absolute_path(cfg.paths.train_audio_dir))
    test_audio_dir = Path(hydra.utils.to_absolute_path(cfg.paths.test_audio_dir))
    cache_dir = Path(hydra.utils.to_absolute_path(cfg.paths.cache_dir))
    train_meta_pickle = Path(hydra.utils.to_absolute_path(cfg.data.train_meta_pickle))
    test_meta_pickle = Path(hydra.utils.to_absolute_path(cfg.data.test_meta_pickle))
    labels_pickle = Path(hydra.utils.to_absolute_path(cfg.paths.labels_pickle))

    extractor = ASTFeatureExtractor.from_pretrained(cfg.model.model_path)

    trim_config = dict(cfg.preprocessing.trim) if hasattr(cfg, "preprocessing") else None

    if cfg.split == "train":
        df_train = pd.read_csv(train_csv)

        if not labels_pickle.exists():
            label2id = save_labels(df_train, str(labels_pickle))
        else:
            label2id = load_labels(str(labels_pickle))

        train_samples = build_samples_list(
            df=df_train,
            cache_dir=cache_dir,
            label2id=label2id,
            split="train",
        )

        extract_and_cache_features(
            df=df_train,
            samples=train_samples,
            audio_dir=train_audio_dir,
            extractor=extractor,
            target_sample_rate=cfg.audio.sample_rate,
            trim_config=trim_config,
        )

        with train_meta_pickle.open("wb") as f:
            pickle.dump(train_samples, f)

    elif cfg.split == "test":
        submission_csv = Path(hydra.utils.to_absolute_path(cfg.paths.submission_csv))
        df_test = pd.read_csv(submission_csv)

        test_samples = build_samples_list(
            df=df_test,
            cache_dir=cache_dir,
            label2id=None,
            split="test",
        )

        extract_and_cache_features(
            df=df_test,
            samples=test_samples,
            audio_dir=test_audio_dir,
            extractor=extractor,
            target_sample_rate=cfg.audio.sample_rate,
            trim_config=trim_config,
        )

        with test_meta_pickle.open("wb") as f:
            pickle.dump(test_samples, f)
    else:
        raise ValueError(f"Unknown split: {cfg.split}")


if __name__ == "__main__":
    main()
