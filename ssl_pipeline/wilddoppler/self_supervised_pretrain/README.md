# SSL Pretraining

Scripts to train backbone encoders from scratch on **unlabeled** micro-Doppler tracks, before any supervised fine-tuning. Two pretraining objectives are provided:

| Method | Script | Description |
|---|---|---|
| Contrastive | `main_contrastive.py` | Multi-positive Supervised Contrastive loss (SupCon). Multiple global and local views are generated per track; all views sharing the same track ID are treated as positives and the encoder is trained to bring them together while pushing other tracks apart. |
| Reconstruction | `main_reconstruction.py` | Masked autoencoder: a fraction of the spectrogram is masked and the model learns to reconstruct the missing patches via a lightweight Transformer decoder. |

---

## Data

Pretraining uses an **unlabeled** tracklist CSV and a directory of `.npz` tracks (same format as the labeled data). Both `--tracklist-csv` and `--data-dir` are required arguments.

Expected layout:

```
DopplerWild/
└── data/
    ├── unlabeled_tracks_Doppler/            # .npz files for pretraining
    └── DopplerWild_unlabeled_tracklist.csv  # columns: file_name, track_id, split
```

The `split` column must be present and contain `"train"` (and optionally `"val"`) rows.

---

## Running pretraining

Run from the repo root as a module so Python resolves the `self_supervised_pretrain` package correctly.

**Contrastive pretraining:**

```bash
python -m self_supervised_pretrain.main_contrastive \
  --model-name mobilenet_v2 \
  --tracklist-csv data/fold_splits/DopplerWild_unlabeled_tracklist.csv \
  --data-dir data/unlabeled_tracks_Doppler
```

**Reconstruction-based pretraining:**

```bash
python -m self_supervised_pretrain.main_reconstruction \
  --model-name mobilenet_v2 \
  --tracklist-csv data/fold_splits/DopplerWild_unlabeled_tracklist.csv \
  --data-dir data/unlabeled_tracks_Doppler
```

Both scripts accept `--model-name resnet18` to train the ResNet-18 variant instead.

---

## Key arguments

**Shared:**

| Argument | Description |
|---|---|
| `--tracklist-csv` | *(required)* Path to unlabeled tracklist CSV |
| `--data-dir` | *(required)* Directory containing `.npz` track files |
| `--model-name` | Backbone: `mobilenet_v2` (default) or `resnet18` |
| `--epochs` | Training epochs |
| `--batch-size` | Batch size |
| `--learning-rate` | LR (default: 1e-4; contrastive also exposes `--backbone-learning-rate` / `--head-learning-rate`) |
| `--cache-mode` | `preload` loads all tracks into RAM at startup; `lazy` caches on first access; `none` reads from disk each time |
| `--ckpt-dir` | Where checkpoints are saved (defaults to a method-specific subdirectory) |

**Contrastive-specific:**

| Argument | Description |
|---|---|
| `--num-global-views` / `--num-local-views` | Views per sample |
| `--temperature` | SupCon softmax temperature |
| `--proj-dim` | Projection head output dimension |

**Reconstruction-specific:**

| Argument | Description |
|---|---|
| `--mask-ratio` | Fraction of spectrogram patches masked |
| `--decoder-dim` / `--decoder-layers` | Transformer decoder size |
| `--mid-focus-prob` | Probability of biasing masking toward mid-frequency band |

---

## Augmentations (contrastive)

The `RadarAugmentations` class in [augmentations.py](augmentations.py) applies a radar-specific augmentation pipeline to each view:

- **Gaussian noise** — additive white noise scaled by `--gaussian-noise-std`, applied to every view
- **Time/frequency masking** — randomly zeros short strips along the time or frequency axis
- **Patch masking** — zeros small rectangular regions
- **Time shift** — circularly shifts the spectrogram along the time axis
- **Time flip** — reverses the time axis with probability `--flip-time-prob`
- **Frequency flip** — mirrors the Doppler axis (simulates sign-ambiguity)
- **Interference mixing** — blends a small fraction of another sample to simulate multi-person scenes
- **Local limb augmentation** (local views only) — enhances high-frequency limb returns via Gaussian decomposition and velocity normalization

---

## Output checkpoints

Checkpoints are saved to `--ckpt-dir` (defaults: `self_supervised_pretrain/checkpoints/contrastive/` and `self_supervised_pretrain/checkpoints/reconstruction/`). Per-epoch snapshots are named `<method>_<model_name>_epoch<N>.pt`; the best validation checkpoint is `<method>_<model_name>_best.pt`. Pass the best checkpoint to the downstream fine-tuning step via `paths.ckpt` (see the main [README](../README.md)).
