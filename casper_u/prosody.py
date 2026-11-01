"""Extract utterance-level prosodic and spectral descriptors from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000
HOP = 160
N_FFT = 512


def stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    if not values.size:
        return [0.0] * 6
    return [
        float(values.mean()),
        float(values.std()),
        float(np.median(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
        float(values.max() - values.min()),
    ]


def one(path: str | Path) -> np.ndarray:
    """Compute the same descriptor layout for ordinary and very short recordings."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise ValueError(f"Empty audio file: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    # Prevent invalid analysis windows on extremely short clips; retain true duration.
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=400, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP
    )[0]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP)[
        0
    ]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=20, n_fft=N_FFT, hop_length=HOP
    )
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
    delta = (
        librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    )
    f0 = librosa.yin(
        audio, fmin=60, fmax=400, sr=SAMPLE_RATE, frame_length=400, hop_length=HOP
    )
    f0 = np.where(np.isfinite(f0) & (f0 >= 60) & (f0 <= 400), f0, 0.0)
    voiced = f0[f0 > 0]
    jitter = (
        float(np.abs(np.diff(voiced)).mean() / (voiced.mean() + 1e-6))
        if len(voiced) > 1
        else 0.0
    )
    voiced_ratio = float(len(voiced) / max(len(f0), 1))
    shimmer = (
        float(np.abs(np.diff(rms)).mean() / (rms.mean() + 1e-6))
        if len(rms) > 1
        else 0.0
    )
    features = stats(voiced) + [jitter, voiced_ratio] + stats(rms) + [shimmer]
    for descriptor in (zcr, centroid, bandwidth, rolloff, flatness):
        features.extend(stats(descriptor))
    features.extend(mfcc.mean(axis=1).tolist())
    features.extend(mfcc.std(axis=1).tolist())
    features.extend(delta.mean(axis=1).tolist())
    features.extend(delta.std(axis=1).tolist())
    features.append(duration)
    result = np.asarray(features, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite prosodic descriptors for: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.dataset).name != args.dataset or not args.dataset:
        parser.error("dataset must be a corpus name, not a path")
    manifest = args.manifest_dir.expanduser().resolve() / f"{args.dataset}.csv"
    frame = pd.read_csv(manifest, dtype={"utt_id": str, "path": str})
    missing = {"utt_id", "path"} - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty or frame.utt_id.isna().any() or frame.path.isna().any():
        raise ValueError(
            "Manifest must contain non-empty utterance identifiers and paths"
        )
    if frame.utt_id.duplicated().any():
        raise ValueError("Manifest contains duplicate utterance identifiers")
    destination = args.feature_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{args.dataset}_prosody.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output}")
    rows = []
    for raw_path in tqdm(frame.path, desc=args.dataset, unit="utterance"):
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        rows.append(one(path))
    matrix = np.stack(rows, axis=0)
    np.savez(output, X=matrix, utt_ids=np.asarray(frame.utt_id, dtype=str))
    print(f"Saved {output}: {matrix.shape}")


if __name__ == "__main__":
    main()
