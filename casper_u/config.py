"""Main-model settings derived from declared corpus design metadata."""

import argparse
from pathlib import Path

UNIFIED = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]
TARGET_CLASSES = {
    "EmoDB": ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"],
    "CASIA": ["anger", "fear", "happiness", "neutral", "sadness", "surprise"],
    "BodEMODB": ["anger", "fear", "happiness", "neutral", "sadness", "surprise"],
    "IEMOCAP": ["anger", "happiness", "neutral", "sadness"],
}
# Corpus-level declarations, supplied independently of per-recording test labels.
# Change these declarations when using a different recording design or subset.
CORPUS_DESIGN = {
    "EmoDB": dict(
        speakers=10,
        utterances=535,
        imbalance=2.76,
        parallel_fraction=0.02,
        parallel_groups=2,
        dialogues=1,
        average_turns=1.0,
    ),
    "CASIA": dict(
        speakers=4,
        utterances=1200,
        imbalance=1.00,
        parallel_fraction=1.0,
        parallel_groups=200,
        dialogues=1,
        average_turns=1.0,
    ),
    "BodEMODB": dict(
        speakers=10,
        utterances=3000,
        imbalance=1.00,
        parallel_fraction=1.0,
        parallel_groups=500,
        dialogues=1,
        average_turns=1.0,
    ),
    "IEMOCAP": dict(
        speakers=10,
        utterances=4485,
        imbalance=2.90,
        parallel_fraction=0.002,
        parallel_groups=1,
        dialogues=151,
        average_turns=29.7,
    ),
}


def main_settings(target, datasets):
    info = CORPUS_DESIGN[target]
    external = [set(TARGET_CLASSES[d]) for d in datasets if d != target]
    target_labels = set(TARGET_CLASSES[target])
    context = info["dialogues"] >= 10 and info["average_turns"] >= 2
    return dict(
        pspc=info["parallel_fraction"] >= 0.95 and info["parallel_groups"] >= 10,
        iemo_context=context,
        iemo_prosody=context,
        iemo_context_pca=128 if context else 64,
        iemo_prosody_pca=32,
        adv_weight=0.2 if info["speakers"] <= 4 else 0.0,
        no_class_weights=not (info["imbalance"] >= 1.05 or info["utterances"] < 2000),
        ext_weight=(
            0.0 if not external or all(target_labels < e for e in external) else 0.1
        ),
        # The conversational main preset uses a shared output inventory.
        single_head=target == "IEMOCAP",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CASPER-U main-model LOSO pipeline")
    parser.add_argument("--target", required=True, choices=tuple(TARGET_CLASSES))
    parser.add_argument("--datasets", default="EmoDB,CASIA,BodEMODB,IEMOCAP")
    parser.add_argument("--manifest-dir", type=Path, default=Path("work/manifests"))
    parser.add_argument("--feature-dir", type=Path, default=Path("work/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", dest="bs", type=int, default=128)
    parser.add_argument("--per-layer-pca", dest="per_layer_pca", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", dest="wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", dest="ls", type=float, default=0.1)
    parser.add_argument("--avg-last", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    if (
        len(set(datasets)) != len(datasets)
        or any(d not in TARGET_CLASSES for d in datasets)
        or args.target not in datasets
    ):
        parser.error(
            "--datasets must contain unique supported names including the target"
        )
    if min(args.epochs, args.bs, args.per_layer_pca) < 1 or args.hidden < 2:
        parser.error(
            "epochs, batch size, PCA size must be positive and hidden width at least 2"
        )
    if not 1 <= args.avg_last <= args.epochs:
        parser.error("--avg-last must lie between 1 and --epochs")
    if (
        not 0 <= args.dropout < 1
        or not 0 <= args.ls <= 1
        or args.lr <= 0
        or args.wd < 0
    ):
        parser.error("invalid optimizer or regularization settings")
    for key, value in main_settings(args.target, datasets).items():
        setattr(args, key, value)
    args.layers = "0,4,8,12,16,20,24,28"
    args.target_weight = 10.0
    args.pspc_lambda = 0.5
    args.tag = "main"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args
