# Implementation notes

## Representations and preprocessing

The main model concatenates the temporal mean and standard deviation within each selected encoder layer. Each combined layer block is standardized and projected to 128 PCA components. Eight projected blocks form a 1024-dimensional acoustic vector with the default settings.

Speaker centering subtracts a separate mean for each corpus-qualified speaker. It uses unlabeled recordings from both training speakers and the held-out speaker. StandardScaler and PCA are then fitted on the concatenation of the target training speakers and external corpora. External recordings remain in this preprocessing pool when their supervised loss weight is zero.

## Network and loss

The shared trunk uses LayerNorm, a linear layer of width 512, GELU, dropout, a second linear layer of width 256, GELU, and dropout. Classification layers map that shared representation to the configured output inventories.

Target samples have supervised loss weight 10. External samples have weight `external_weight * target_training_count / external_count`. Samples are shuffled through the combined training pool; corpus balancing is implemented through loss weights. Class weights are computed from the training labels of each output inventory.

The CASIA preset adds a speaker classifier. Its coefficient 0.2 is applied both to gradient reversal and to the speaker loss, so the contribution to the trunk's adversarial gradient contains both factors.

The IEMOCAP preset uses one shared eight-class classification layer. Its predicted probabilities are restricted to the four target classes and renormalized before evaluation. The other presets use separate corpus classification layers.

## Context and prosody layout

For the conversational target, three leave-current-utterance-out dialogue summaries are computed in the projected representation space: all other utterances, the same speaker, and the other speaker. The context block is standardized and projected using target training rows. Prosodic descriptors are processed in the same way.

The current feature layout reserves additional zero coordinates and writes projected context and prosody into leading coordinates of the conversational vectors, replacing those acoustic coordinates. The trailing reserved coordinates stay zero. This layout is retained by the main training implementation; it is not an appended-feature layout.

## Structured prediction

PSPC solves a maximum-score one-to-one assignment in a complete `(speaker, sentence)` group with as many recordings as emotion classes. The calibrated distribution blends independent probabilities and the assigned one-hot vectors with coefficient 0.5. An incomplete group falls back to its independent probability distribution.

The design declarations in `config.py` determine whether the component is enabled. They must correspond to the user's actual recording design and subset. The declaration is not a check of test emotion labels, and the group-size check cannot verify that an incorrectly declared group contains one recording from each emotion.

## Runtime behavior

Training uses CPU tensors. Feature extraction has its own device and dtype options. The extractor and trainer have different memory costs, and feature extraction is performed separately for each corpus.

LOSO probabilities are averaged over the last five epochs by default. No epoch is selected by held-out emotion labels. Emotion labels for the held-out speaker are used to compute evaluation metrics after prediction.

## Checked environment

The audio utilities and a locally initialized small Whisper encoder were checked with Python 3.12, PyTorch 2.13.0, Transformers 5.14.1, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.18.0, librosa 0.11.0, and SoundFile 0.14.0. No pretrained model download is required for that structural check. Runtime dependencies are declared in `pyproject.toml`; other supported versions may differ numerically.
