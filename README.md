# Simulation_SSL

Simulation and self-supervised learning preparation code for radar micro-Doppler
generation experiments.

This repository contains two connected pieces:

- **Simulation production code** in `code/`: scripts for generating synthetic
  micro-Doppler spectra from SMPL-X / MMRadPose / M4Human style motion inputs.
- **WildDoppler SSL pipeline** in `ssl_pipeline/wilddoppler/`: AMASS generation
  scripts, manifest conversion, self-supervised pretraining, downstream
  fine-tuning, and evaluation utilities.

Large datasets, generated results, caches, checkpoints, and SMPL-X model files
are intentionally not committed.

## Repository Layout

```text
.
├── code/
│   ├── smpl_mesh_to_micro_doppler.py
│   ├── mmradpose_smplx_to_micro_doppler.py
│   ├── mmradpose_raw_to_doppler_time.py
│   ├── mmradpose_sim_echo_compare_gt.py
│   ├── m4human_smplx_to_micro_doppler.py
│   └── run_*.sh
└── ssl_pipeline/
    └── wilddoppler/
        ├── code/smpl_mesh_to_micro_doppler.py
        ├── scripts/amass_generation/
        ├── self_supervised_pretrain/
        ├── ssl_eval/
        ├── supervised/
        ├── conf/
        └── utils/
```

## External Inputs

Prepare these outside Git, preferably on SSD for generation throughput:

```text
AMASS_SMPLX_2022/
smpl_models/
witwin/                         # optional local WiTwin radar package copy
results/                        # generated outputs, mesh cache, profiles
```

The accelerated AMASS generation experiments used this layout:

```bash
ROOT=/ssdtemp/users/$USER/wilddoppler_ssd
```

Expected key paths under `ROOT`:

```text
$ROOT/code/smpl_mesh_to_micro_doppler.py
$ROOT/scripts/amass_generation/
$ROOT/legacy/datasets/AMASS_SMPLX_2022/
$ROOT/smpl_models/
$ROOT/witwin/radar/              # optional, if overriding installed WiTwin
```

## Environment

Install the Python dependencies needed by the selected pipeline. The
WildDoppler SSL dependencies are listed in:

```text
ssl_pipeline/wilddoppler/requirements.txt
```

For RTX Blackwell GPUs, use a PyTorch build that supports the GPU at runtime.
On the tested server, the older CUDA 12.1 PyTorch environment could not run on
the RTX PRO 6000 Blackwell GPU, so a CUDA 12.8 PyTorch environment was used.

## AMASS Generation

The main manifest runner is:

```text
ssl_pipeline/wilddoppler/scripts/amass_generation/run_generation_manifest.py
```

It launches the generator on each manifest row and manages CUDA/cache/Slang
environment variables. The runner includes the accelerated Slang fast-f32 path:

- `WITWIN_SLANG_FRAME_FLOAT32=1`
- `WITWIN_SLANG_BATCH_PREP=1`
- batched chirp sample preparation
- batched total-length/amplitude preparation
- optional RTX Blackwell PTX fallback via `--rtx-blackwell-ptx`

Example cache-hit generation command on RTX Blackwell:

```bash
ROOT=/ssdtemp/users/$USER/wilddoppler_ssd

$ROOT/venv_rtx_sm120/bin/python \
  $ROOT/scripts/amass_generation/run_generation_manifest.py \
  --manifest-csv scripts/amass_generation/amass_equal_target_manifest.csv \
  --workspace-root $ROOT \
  --witwin-python $ROOT/venv_rtx_sm120/bin/python \
  --generator code/smpl_mesh_to_micro_doppler.py \
  --model-dir smpl_models \
  --out-dir results/amass_smplx_micro_doppler/equal_target_raw \
  --log-dir results/amass_smplx_micro_doppler/equal_target_logs \
  --cuda-visible-devices 0 \
  --visibility-mode hold \
  --smpl-batch-size 64 \
  --mesh-cache-dir results/amass_smplx_micro_doppler/mesh_cache \
  --skip-plot \
  --rtx-blackwell-ptx
```

For RTX 3090 / Ampere, use Slang fast-f32 with the native architecture:

```bash
--backend slang \
--enable-slang-fast-f32 \
--torch-cuda-arch-list 8.6 \
--slang-cuda-arch 8.6
```

For Blackwell with CUDA toolchains that cannot compile `sm_120` directly, use:

```bash
--rtx-blackwell-ptx
```

This sets `9.0+PTX` so the driver can JIT PTX for the GPU.

## Profiling One AMASS Sequence

Use:

```text
ssl_pipeline/wilddoppler/scripts/amass_generation/profile_one_amass_generation.py
```

Recommended profiling pattern:

1. Run once with an empty mesh cache to measure cache miss / SMPL-X mesh build.
2. Run again with the same mesh cache to measure cache hit generation.
3. Compare effective time excluding radar/scene/tracer initialization.

Important timing components reported by the profiler:

- SMPL-X mesh generation or mesh cache read
- Radar / scene / tracer initialization
- visibility trace and mesh update
- MIMO chirp simulation
- range FFT and bin selection
- Doppler STFT
- serialization and NPZ write

## SSL Pretraining

SSL code lives in:

```text
ssl_pipeline/wilddoppler/self_supervised_pretrain/
```

Two pretraining objectives are provided:

- contrastive pretraining: `main_contrastive.py`
- masked reconstruction pretraining: `main_reconstruction.py`

Run from `ssl_pipeline/wilddoppler` so imports resolve cleanly:

```bash
cd ssl_pipeline/wilddoppler

python -m self_supervised_pretrain.main_contrastive \
  --model-name mobilenet_v2 \
  --tracklist-csv data/fold_splits/DopplerWild_unlabeled_tracklist.csv \
  --data-dir data/unlabeled_tracks_Doppler
```

Reconstruction:

```bash
python -m self_supervised_pretrain.main_reconstruction \
  --model-name mobilenet_v2 \
  --tracklist-csv data/fold_splits/DopplerWild_unlabeled_tracklist.csv \
  --data-dir data/unlabeled_tracks_Doppler
```

Downstream SSL evaluation and fine-tuning utilities are under:

```text
ssl_pipeline/wilddoppler/ssl_eval/
ssl_pipeline/wilddoppler/supervised/
```

## Notes

- Generated `.npz`, `.npy`, checkpoints, datasets, and cache directories should
  remain outside Git.
- SMPL-X model files are required at runtime but are not committed here.
- Use SSD-backed workspaces for AMASS generation; mesh cache read/write is a
  material part of end-to-end throughput.
- If a GPU cannot run the selected PyTorch/Slang architecture, first check
  `torch.cuda.get_device_name()`, PyTorch CUDA version, and the generated Slang
  `-gencode` flags.
