# Simulator Run Guide

This repository includes the WiTwin radar/core source files needed by the
simulation entry points, including the Slang kernels used by the accelerated
backend.

## Slang Kernel Locations

The kernels were copied from the local WiTwin environment:

```text
/bigdata/users/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar/solvers/radar.slang
/bigdata/users/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar/solvers/dirichlet.slang
/bigdata/users/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/core/mesh_sdf.slang
```

They are now vendored in the repository at:

```text
witwin/radar/solvers/radar.slang
witwin/radar/solvers/dirichlet.slang
witwin/core/mesh_sdf.slang
```

Only source files are tracked. SlangTorch build outputs, `.so` files, lock
files, `__pycache__`, and local backup files are intentionally ignored.

## Environment

Use a Python environment with CUDA-capable PyTorch and the packages listed in
`requirements.txt`.

Fresh clone setup:

```bash
# Recommended when Git LFS is available.
git lfs install
git clone git@github.com:quanshengjie604-beep/Simulation_SSL.git
cd Simulation_SSL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA PyTorch build that matches the host driver/toolkit first.
# Then install the repository runtime dependencies.
python -m pip install -r requirements.txt
```

If the host does not have `git-lfs`, clone the source tree without resolving
LFS objects:

```bash
git -c filter.lfs.required=false \
  -c filter.lfs.smudge= \
  -c filter.lfs.process= \
  clone git@github.com:quanshengjie604-beep/Simulation_SSL.git
cd Simulation_SSL
```

The tested local interpreter on the development server is:

```bash
/bigdata/users/quansj/miniforge3/envs/witwin/bin/python
```

Run commands from the repository root. The scripts now prefer the vendored
WiTwin package in `witwin/` when present, so a separate WiTwin checkout is not
required. You can still force a different WiTwin checkout or installation with:

```bash
export WITWIN_RADAR_DIR=/path/to/witwin/radar
```

Keep compiler caches on writable local storage:

```bash
export WITWIN_CACHE_DIR=/tmp/witwin-cache
export XDG_CACHE_HOME=$WITWIN_CACHE_DIR
export DRJIT_CACHE_DIR=$WITWIN_CACHE_DIR
```

Required external data and model files are not shipped in Git:

```text
datasets/AMASS_SMPLX_2022/       # AMASS SMPL-X motion files
smpl_models/                     # SMPL/SMPL-X model files accepted by smplx
results/                         # generated outputs and mesh cache
```

Before a full run, check the repository and Python environment:

```bash
test -f witwin/radar/solvers/radar.slang
test -f witwin/radar/solvers/dirichlet.slang
test -f witwin/core/mesh_sdf.slang

python -m py_compile \
  code/smpl_mesh_to_micro_doppler.py \
  ssl_pipeline/wilddoppler/code/smpl_mesh_to_micro_doppler.py

python - <<'PY'
import torch
import slangtorch
import smplx
import drjit
import mitsuba
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("slangtorch import ok")
PY
```

## Minimal AMASS Simulation

The direct SMPL-X/AMASS generator is:

```text
code/smpl_mesh_to_micro_doppler.py
```

Example Slang run:

```bash
ROOT=$PWD
PYTHON=/bigdata/users/quansj/miniforge3/envs/witwin/bin/python

$PYTHON code/smpl_mesh_to_micro_doppler.py \
  --amass-file datasets/AMASS_SMPLX_2022/BMLmovi/Subject_11_F_MoSh/Subject_11_F_5_stageii.npz \
  --model-dir smpl_models \
  --out results/example_micro_doppler.npz \
  --backend slang \
  --device cuda:0 \
  --enable-slang-fast-f32 \
  --skip-plot
```

For RTX 3090 / Ampere, add:

```bash
--torch-cuda-arch-list 8.6 \
--slang-cuda-arch 8.6
```

For RTX Blackwell systems where the CUDA toolchain cannot compile native
`sm_120` objects, use the PTX fallback:

```bash
--rtx-blackwell-ptx
```

## Manifest-Based AMASS Generation

The WildDoppler manifest runner is:

```text
ssl_pipeline/wilddoppler/scripts/amass_generation/run_generation_manifest.py
```

Example:

```bash
ROOT=$PWD
PYTHON=/bigdata/users/quansj/miniforge3/envs/witwin/bin/python

$PYTHON ssl_pipeline/wilddoppler/scripts/amass_generation/run_generation_manifest.py \
  --manifest-csv ssl_pipeline/wilddoppler/scripts/amass_generation/amass_equal_target_manifest.csv \
  --workspace-root $ROOT \
  --witwin-python $PYTHON \
  --generator code/smpl_mesh_to_micro_doppler.py \
  --model-dir smpl_models \
  --out-dir results/amass_smplx_micro_doppler/equal_target_raw \
  --log-dir results/amass_smplx_micro_doppler/equal_target_logs \
  --cuda-visible-devices 0 \
  --visibility-mode hold \
  --smpl-batch-size 64 \
  --mesh-cache-dir results/amass_smplx_micro_doppler/mesh_cache \
  --backend slang \
  --enable-slang-fast-f32 \
  --skip-plot
```

## Outputs

Generated `.npz` files and plots should stay under `results/` or another
external output directory. They are runtime artifacts and should not be
committed to Git.
