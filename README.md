# CASPER-U

CASPER-U is a speech emotion recognition framework built around cross-layer Whisper representations, speaker normalization, a shared prediction network, and corpus-conditioned output layers. It combines acoustic representations with recording structure through parallel-sentence permutation calibration (PSPC) and dialogue-context features.

## Main workflow

1. Build or supply utterance manifests containing audio paths, speaker identities, emotion labels, sentence identifiers, and dialogue identifiers.
2. Extract temporal means and standard deviations from Whisper encoder layers.
3. Center representations by speaker, then fit standardization and per-layer PCA within each training fold.
4. Train the shared predictor with the main corpus configuration and generate predictions for each held-out speaker.
5. Apply PSPC when the recording design specifies one recording per emotion within each speaker–sentence group.

The training entry point runs leave-one-speaker-out (LOSO) evaluation. Speaker normalization uses the unlabeled recordings of the held-out speaker. Dialogue summaries use the complete interaction, including other turns from that speaker; the conversational partner can remain in the training split. PSPC uses the declared recording design and group identifiers, without accessing individual test emotion labels.

## Installation

Use Python 3.10 or newer and install the project in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Acquire the audio corpora and a Hugging Face-compatible Whisper-large-v3 model separately. Keep them outside the repository or inside the ignored `work/` directory.

## Prepare manifests

The included builder supports these local directory layouts:

```text
<audio-root>/EmoDB/<emotion>/*.wav
<audio-root>/BodEMODB/<emotion>/*.wav
<audio-root>/CASIA/<speaker>/<emotion>/*.wav
<audio-root>/IEMOCAP/<emotion>/*.wav
```

```bash
python -m casper_u.manifests --data-root /path/to/audio-root \
  --manifest-dir work/manifests
```

For a different directory layout, supply your own per-corpus CSV manifests. Training requires `utt_id`, `label`, `speaker`, `sentence`, and `dialogue`. Audio extraction also needs `path`; supplying `duration_s` preserves duration-based batching. Relative audio paths are resolved from the manifest directory. Use unique utterance IDs and genuine sentence/dialogue identifiers from the recording design. The supported label inventories are defined in `casper_u/config.py`.

The input files are named `EmoDB.csv`, `CASIA.csv`, `BodEMODB.csv`, and `IEMOCAP.csv`. Manifests are local input files and are not included in this repository.

## Extract representations

Run extraction for each corpus participating in the training pool:

```bash
for corpus in EmoDB CASIA BodEMODB IEMOCAP; do
  python -m casper_u.features --dataset "$corpus" \
    --manifest-dir work/manifests --feature-dir work/features \
    --model /path/to/whisper-large-v3 --device auto --local-files-only
done
```

Extraction uses dynamically padded batches, pools only valid encoder frames, and truncates recordings at the encoder input window. Auto device selection prefers CUDA, then MPS, then CPU; CPU inference uses float32 by default. Each cache contains `mean`, `std`, and `utt_ids`. The feature arrays have shape `(utterances, encoder_layers, encoder_width)`. Utterance IDs align the cache with its manifest. Existing caches using these fields and the filename `<corpus>_whisperL.npz` can also be used.

The conversational preset also uses prosodic descriptors:

```bash
python -m casper_u.prosody --dataset IEMOCAP \
  --manifest-dir work/manifests --feature-dir work/features
```

## Run the main model

```bash
python -m casper_u.train --target EmoDB
python -m casper_u.train --target BodEMODB
python -m casper_u.train --target CASIA
python -m casper_u.train --target IEMOCAP
```

The default training pool includes all four corpora. Use `--manifest-dir`, `--feature-dir`, and `--output-dir` to select local input and output directories. The `--datasets` option accepts a comma-separated list when only a subset of the corpora is available; the target must be included. Changing the training pool changes the evaluation setting.

Main configuration defaults are:

- Encoder layers: `0, 4, 8, 12, 16, 20, 24, 28`, indexed from zero.
- A 128-component PCA projection for each concatenated mean/std layer block.
- A shared network with hidden widths 512 and 256, GELU, LayerNorm, and dropout.
- AdamW, 30 epochs, and an average of predicted probabilities over the final 5 epochs.
- Corpus-level metadata rules for PSPC, context/prosody, speaker adversarial learning, class weighting, and external supervision.

The conversational main preset uses a shared eight-class output inventory, mapped to its four target emotions, and a 128-component context projection. Other presets use corpus-specific classification layers. See [implementation notes](docs/implementation.md) for the precise feature layout and loss weighting.

The run writes a JSON summary and per-fold prediction probabilities to `work/outputs/`. `uar_mean` is macro Recall averaged over speaker folds; `war_mean` is Accuracy averaged over folds. The entry point evaluates and exports predictions; it does not save a standalone deployment checkpoint with fitted preprocessing.

## Source layout

| Module | Role |
|---|---|
| `casper_u/config.py` | Label inventories, declared design metadata, main-model settings |
| `casper_u/manifests.py` | Local audio metadata preparation |
| `casper_u/features.py` | Whisper encoder statistics |
| `casper_u/prosody.py` | Prosodic and spectral descriptors |
| `casper_u/model.py` | Shared network, classification layers, gradient reversal |
| `casper_u/utils.py` | Speaker centering, PSPC assignment, random seeds |
| `casper_u/train.py` | Main LOSO training and prediction workflow |

Audio data, manifests, pretrained weights, feature caches, and generated outputs remain local. The repository's ignore rules allow source files and project documentation only.
