import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 16kHz log-mel features and save to pickle.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument("--csv-path", type=Path, default=None, help="CSV file with columns: fname,label")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory with wav files")
    parser.add_argument("--output-pickle", type=Path, default=None, help="Output pickle path")
    parser.add_argument("--labels-pickle", type=Path, default=None, help="Optional output label mapping pickle")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--log-offset", type=float, default=None)
    parser.add_argument("--duration-sec", type=float, default=None, help="Target clip duration for frame count")
    parser.add_argument("--fmin-hz", type=float, default=None)
    parser.add_argument("--fmax-hz", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None, help="Number of process workers")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def resolve_settings(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    data_cfg = cfg.get("data", {})
    audio_cfg = cfg.get("audio", {})

    if args.split == "train":
        args.csv_path = args.csv_path or Path(data_cfg["train_csv"])
        args.audio_dir = args.audio_dir or Path(data_cfg["train_audio_dir"])
        args.output_pickle = args.output_pickle or Path(data_cfg["train_pickle"])
        args.output_dir = Path(data_cfg["train_np_dir"])
        args.labels_pickle = args.labels_pickle or Path(data_cfg.get("labels_pickle", "data/labels.pickle"))
    else:
        args.audio_dir = args.audio_dir or Path(data_cfg["test_audio_dir"])
        args.output_pickle = args.output_pickle or Path(data_cfg["test_pickle"])
        args.output_dir = Path(data_cfg["test_np_dir"])
        if args.labels_pickle is None and data_cfg.get("labels_pickle"):
            args.labels_pickle = Path(data_cfg["labels_pickle"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.sample_rate = args.sample_rate or int(audio_cfg.get("sample_rate", 16000))
    args.n_fft = args.n_fft or int(audio_cfg.get("n_fft", 512))
    args.hop_length = args.hop_length or int(audio_cfg.get("hop_length", 160))
    args.n_mels = args.n_mels or int(audio_cfg.get("n_mels", 64))
    args.log_offset = args.log_offset or float(audio_cfg.get("log_offset", 1e-6))
    args.duration_sec = args.duration_sec or float(audio_cfg.get("duration_sec", 2.0))
    args.fmin_hz = args.fmin_hz or float(audio_cfg.get("fmin_hz", 0))
    args.fmax_hz = args.fmax_hz or float(audio_cfg.get("fmax_hz", args.sample_rate / 2))
    return args


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "fname" not in df.columns or "label" not in df.columns:
        if df.shape[1] >= 2:
            df = pd.read_csv(csv_path, skiprows=1, names=["fname", "label"])
        else:
            raise ValueError("CSV must contain fname and label columns.")
    return df[["fname", "label"]]


def build_unlabeled_metadata(audio_dir: Path) -> pd.DataFrame:
    filenames = sorted(
        path.name
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    )
    if not filenames:
        raise ValueError(f"No supported audio files found in {audio_dir}")
    return pd.DataFrame({"fname": filenames})


def extract_log_mel(
    wav_path: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    log_offset: float,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    waveform, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
        power=2.0,
    )
    log_mel = np.log(mel + log_offset)
    return log_mel.astype(np.float32, copy=False)


def process_single_file(task_data, common_args):
    fname, label_id = task_data
    try:
        wav_path = Path(common_args.audio_dir) / fname

        feature = extract_log_mel(
            wav_path=wav_path,
            sample_rate=common_args.sample_rate,
            n_fft=common_args.n_fft,
            hop_length=common_args.hop_length,
            n_mels=common_args.n_mels,
            log_offset=common_args.log_offset,
            fmin_hz=common_args.fmin_hz,
            fmax_hz=common_args.fmax_hz,
        )

        npy_name = Path(fname).with_suffix(".npy").name
        output_path = common_args.output_dir / npy_name
        np.save(output_path, feature)

        return {"fname": fname, "npy_path": str(output_path), "label_id": label_id}, None
    except Exception as exc:
        return None, f"{fname}: {exc}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args = resolve_settings(args, cfg)
    if args.split == "train":
        if args.csv_path is None:
            raise ValueError("Training preprocessing requires --csv-path or data.train_csv in config.")
        metadata = load_metadata(args.csv_path)
        labels = sorted(metadata["label"].unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
    else:
        metadata = build_unlabeled_metadata(args.audio_dir)
        label_to_id = None

    tasks = []
    for row in metadata.itertuples(index=False):
        label_id = label_to_id[row.label] if label_to_id else None
        tasks.append((row.fname, label_id))

    worker_fn = partial(process_single_file, common_args=args)

    max_workers = args.max_workers or (os.cpu_count() or 1)
    ordered_results: list[dict | None] = [None] * len(tasks)
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(worker_fn, task): idx for idx, task in enumerate(tasks)}
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Extracting log-mel features",
        ):
            idx = future_to_index[future]
            result, error = future.result()
            if error is not None:
                errors.append(error)
                continue
            ordered_results[idx] = result

    results = [item for item in ordered_results if item is not None]

    args.output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pickle.open("wb") as f:
        pickle.dump(results, f)

    if label_to_id is not None and args.labels_pickle is not None:
        args.labels_pickle.parent.mkdir(parents=True, exist_ok=True)
        with args.labels_pickle.open("wb") as f:
            pickle.dump(label_to_id, f)

    print(f"Saved {len(results)} items to {args.output_pickle}")
    if errors:
        error_log_path = args.output_pickle.with_suffix(".errors.log")
        error_log_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"Skipped {len(errors)} files due to errors. Details: {error_log_path}")


if __name__ == "__main__":
    main()
