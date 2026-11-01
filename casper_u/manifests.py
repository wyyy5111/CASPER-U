"""Build manifests from supported emotion-folder audio layouts.

Expected layouts below data-root:
  EmoDB/<emotion>/<filename>.wav
  BodEMODB/<emotion>/<speaker><gender><sentence><emotion-code>.wav
  CASIA/<speaker>/<emotion>/<sentence>.wav
  IEMOCAP/<emotion>/Ses01F_impro01_F012.wav

IEMOCAP recordings must already be organized into the selected emotion folders;
this module does not parse the original IEMOCAP annotation files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import soundfile as sf

LABELS = {
    "EmoDB": {"anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"},
    "BodEMODB": {"anger", "fear", "happiness", "neutral", "sadness", "surprise"},
    "CASIA": {"anger", "fear", "happiness", "neutral", "sadness", "surprise"},
    "IEMOCAP": {"anger", "happiness", "neutral", "sadness"},
}
ALIASES = {
    "anger": "anger",
    "angry": "anger",
    "an": "anger",
    "boredom": "boredom",
    "bored": "boredom",
    "disgust": "disgust",
    "fear": "fear",
    "fe": "fear",
    "happiness": "happiness",
    "happy": "happiness",
    "ha": "happiness",
    "neutral": "neutral",
    "ne": "neutral",
    "sadness": "sadness",
    "sad": "sadness",
    "sa": "sadness",
    "surprise": "surprise",
    "su": "surprise",
}
COMMON4 = {"anger", "happiness", "neutral", "sadness"}
COLUMNS = [
    "utt_id",
    "dataset",
    "path",
    "label",
    "label4",
    "speaker",
    "session",
    "dialogue",
    "gender",
    "duration_s",
    "sr",
    "corpus_label",
    "sentence",
]
BOD_NAME = re.compile(r"^(\d{2})([MF])(\d{2})([A-Za-z]{2})\.wav$", re.IGNORECASE)
IEMOCAP_NAME = re.compile(
    r"^(Ses\d{2})([FM])_((?:impro|script)\d+(?:_\d+)?[ab]?)_([FM])(\d+)\.wav$"
)


def normalize_label(value: str) -> str:
    return ALIASES.get(value.strip().lower(), value.strip().lower())


def collect(data_root: Path, dataset: str) -> list[dict]:
    """Read local file metadata without loading or copying waveform samples."""
    directory = data_root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing corpus directory: {directory}")
    pattern = "*/*/*.wav" if dataset == "CASIA" else "*/*.wav"
    rows = []
    for path in sorted(directory.glob(pattern)):
        name = path.name
        emotion = path.parent.name
        if dataset == "BodEMODB":
            match = BOD_NAME.fullmatch(name)
            if not match:
                raise ValueError(f"Unsupported BodEMODB filename: {name}")
            speaker, gender, sentence, emotion = match.groups()
            gender = gender.upper()
            row = dict(
                utt_id=f"{dataset}/{path.stem}",
                speaker=f"{speaker}{gender}",
                session="all",
                dialogue="all",
                gender=gender,
                sentence=sentence,
            )
        elif dataset == "CASIA":
            speaker = path.parent.parent.name
            row = dict(
                utt_id=f"{dataset}/{speaker}/{emotion}/{path.stem}",
                speaker=speaker,
                session="all",
                dialogue="all",
                gender="unknown",
                sentence=path.stem,
            )
        elif dataset == "EmoDB":
            if len(name) < 7:
                raise ValueError(f"Unsupported EmoDB filename: {name}")
            speaker = name[:2]
            # Speaker identities and sentence identities come from filenames.
            row = dict(
                utt_id=f"{dataset}/{path.stem}",
                speaker=speaker,
                session="all",
                dialogue="all",
                gender="M" if speaker in {"03", "08", "09", "10", "11", "12"} else "F",
                sentence=name[2:5],
            )
        else:
            match = IEMOCAP_NAME.fullmatch(name)
            if not match:
                raise ValueError(f"Unsupported IEMOCAP filename: {name}")
            session, session_gender, dialogue, turn_gender, turn = match.groups()
            row = dict(
                utt_id=f"{dataset}/{path.stem}",
                speaker=f"{session}{turn_gender}",
                session=session,
                dialogue=f"{session}{session_gender}_{dialogue}",
                gender=turn_gender,
                sentence=turn,
            )
        label = normalize_label(emotion)
        if label not in LABELS[dataset]:
            raise ValueError(f"Unsupported emotion {emotion!r} in {path}")
        info = sf.info(str(path))
        if info.frames < 1:
            raise ValueError(f"Empty audio file: {path}")
        row.update(
            dataset=dataset,
            path=str(path.resolve()),
            label=label,
            label4=label if label in COMMON4 else "other",
            corpus_label=emotion,
            duration_s=round(info.duration, 4),
            sr=info.samplerate,
        )
        rows.append(row)
    if not rows:
        raise ValueError(
            f"No WAV recordings matched the supported {dataset} layout in {directory}"
        )
    identifiers = [row["utt_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate utterance identifiers in {dataset}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(LABELS),
        help="Corpus names separated by spaces or commas",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    datasets = [name for item in args.datasets for name in item.split(",") if name]
    if not datasets or len(set(datasets)) != len(datasets):
        parser.error("datasets must contain distinct corpus names")
    unsupported = set(datasets) - set(LABELS)
    if unsupported:
        parser.error(f"Unsupported datasets: {sorted(unsupported)}")
    data_root = args.data_root.expanduser().resolve()
    destination = args.manifest_dir.expanduser().resolve()
    outputs = [destination / f"{dataset}.csv" for dataset in datasets] + [
        destination / "all.csv",
        destination / "summary.json",
    ]
    occupied = [path for path in outputs if path.exists()]
    if occupied and not args.overwrite:
        raise FileExistsError(
            f"Outputs already exist; use --overwrite to replace them: {occupied}"
        )
    corpora = {dataset: collect(data_root, dataset) for dataset in datasets}
    destination.mkdir(parents=True, exist_ok=True)
    combined = []
    summary = {}
    for dataset, rows in corpora.items():
        write_csv(destination / f"{dataset}.csv", rows)
        combined.extend(rows)
        summary[dataset] = dict(
            n=len(rows),
            labels=dict(Counter(row["label"] for row in rows)),
            speakers=dict(Counter(row["speaker"] for row in rows)),
            hours=round(sum(row["duration_s"] for row in rows) / 3600, 3),
        )
        print(
            f"{dataset}: {len(rows)} utterances, {len(summary[dataset]['speakers'])} speakers"
        )
    write_csv(destination / "all.csv", combined)
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Saved manifests to {destination}")


if __name__ == "__main__":
    main()
