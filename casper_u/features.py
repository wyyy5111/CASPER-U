"""Extract per-layer Whisper mean and standard-deviation representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

SAMPLE_RATE = 16000
ENCODER_FRAME_SECONDS = 0.02
N_LAYERS = 32


def load_audio(path: str | Path) -> np.ndarray:
    """Load mono audio and resample it to the Whisper input sampling rate."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise ValueError(f"Empty audio file: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def valid_enc_frames(n_samples: int, total_frames: int) -> int:
    """Count valid encoder frames, excluding fixed-window input padding."""
    count = int(np.ceil(n_samples / (SAMPLE_RATE * ENCODER_FRAME_SECONDS)))
    return min(total_frames, max(1, count))


def pool_stats(
    hidden: torch.Tensor, lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool valid frames with the population standard deviation."""
    hidden = hidden.float()
    mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = mask < torch.tensor(lengths, device=hidden.device)[:, None]
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    denominator = mask.sum(dim=1).clamp_min(1)
    mean = (hidden * mask).sum(dim=1) / denominator
    variance = (((hidden - mean[:, None, :]) ** 2) * mask).sum(dim=1) / denominator
    return mean, variance.clamp_min(0).sqrt()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(name)


def extract(args: argparse.Namespace) -> Path:
    """Write mean/std arrays and utterance identifiers for one manifest."""
    from transformers import WhisperFeatureExtractor, WhisperModel

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    manifest = Path(args.manifest_dir).expanduser().resolve() / f"{args.dataset}.csv"
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
    if "duration_s" in frame:
        frame = frame.sort_values("duration_s").reset_index(drop=True)

    feature_dir = Path(args.feature_dir).expanduser().resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)
    output = feature_dir / f"{args.dataset}_whisperL.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output}")

    device = resolve_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16}.get(args.dtype)
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.float16
    extractor = WhisperFeatureExtractor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = WhisperModel.from_pretrained(
        args.model, local_files_only=args.local_files_only, torch_dtype=dtype
    )
    # Only the encoder is needed; release the decoder before moving to an accelerator.
    encoder = model.encoder
    model.decoder = None
    del model
    encoder = encoder.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)
    if len(encoder.layers) != N_LAYERS:
        raise ValueError(
            f"Expected {N_LAYERS} encoder layers, got {len(encoder.layers)}"
        )
    if extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz model input")
    window_samples = int(extractor.n_samples)
    window_frames = int(encoder.embed_positions.weight.shape[0])
    full_positions = encoder.embed_positions.weight.detach().clone()
    dimension = int(encoder.config.d_model)
    means = np.empty((len(frame), N_LAYERS, dimension), dtype=np.float16)
    stds = np.empty_like(means)
    pooled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    lengths: list[int] = []

    def hook_for(layer_index: int):
        def capture(_module, _inputs, layer_output):
            hidden = (
                layer_output[0] if isinstance(layer_output, tuple) else layer_output
            )
            mean, std = pool_stats(hidden, lengths)
            pooled[layer_index] = (
                mean.cpu().numpy().astype(np.float16),
                std.cpu().numpy().astype(np.float16),
            )

        return capture

    hooks = [
        layer.register_forward_hook(hook_for(i))
        for i, layer in enumerate(encoder.layers)
    ]
    start = 0
    progress = tqdm(total=len(frame), desc=args.dataset, unit="utterance")
    try:
        while start < len(frame):
            # Match the duration-sorted batch-size and sample-budget policy.
            stop = start
            budget = 0
            while stop < len(frame) and stop - start < args.batch_size:
                row = frame.iloc[stop]
                if "duration_s" in frame:
                    duration = float(row.duration_s)
                else:
                    path = Path(row.path).expanduser()
                    if not path.is_absolute():
                        path = manifest.parent / path
                    duration = sf.info(str(path)).duration
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError("Manifest durations must be finite and positive")
                sample_count = min(
                    int(np.ceil(duration * SAMPLE_RATE / 320)) * 320, window_samples
                )
                if stop > start and budget + sample_count > args.batch_size * 320 * 400:
                    break
                budget += sample_count
                stop += 1
            subset = frame.iloc[start:stop]
            waveforms = []
            lengths.clear()
            for raw_path in subset.path:
                audio_path = Path(raw_path).expanduser()
                if not audio_path.is_absolute():
                    audio_path = manifest.parent / audio_path
                waveform = load_audio(audio_path)[:window_samples]
                waveforms.append(waveform)
                lengths.append(valid_enc_frames(len(waveform), window_frames))
            padded_samples = int(np.ceil(max(map(len, waveforms)) / 320)) * 320
            frames = padded_samples // 320
            if frames > window_frames:
                raise ValueError("Input exceeds the encoder positional window")
            # Slice actual positional weights as well as updating the configuration.
            # The original full tensor remains available for subsequent longer batches.
            encoder.embed_positions = torch.nn.Embedding.from_pretrained(
                full_positions[:frames], freeze=True
            )
            encoder.config.max_source_positions = frames
            inputs = extractor(
                waveforms,
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                max_length=padded_samples,
                truncation=True,
                return_tensors="pt",
            ).input_features.to(device=device, dtype=dtype)
            if inputs.shape[-1] != frames * 2:
                raise ValueError(
                    "Feature extractor and encoder positional windows do not match"
                )
            pooled.clear()
            with torch.inference_mode():
                encoder(inputs)
            if len(pooled) != N_LAYERS:
                raise RuntimeError("The encoder did not return every expected layer")
            for i in range(N_LAYERS):
                means[start:stop, i] = pooled[i][0]
                stds[start:stop, i] = pooled[i][1]
            start = stop
            progress.update(len(subset))
    finally:
        progress.close()
        encoder.embed_positions = torch.nn.Embedding.from_pretrained(
            full_positions, freeze=True
        )
        encoder.config.max_source_positions = window_frames
        for hook in hooks:
            hook.remove()
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError(
            "Encoder statistics contain non-finite values; try --dtype float32"
        )
    np.savez(output, mean=means, std=stds, utt_ids=np.asarray(frame.utt_id, dtype=str))
    print(f"Saved {output}: mean/std {means.shape}, device={device}, dtype={dtype}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local Whisper model directory or explicit model ID",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.dataset).name != args.dataset or not args.dataset:
        parser.error("dataset must be a corpus name, not a path")
    extract(args)


if __name__ == "__main__":
    main()
