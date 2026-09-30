# TAGSL

Official code release for TAGSL.

This repository provides a runnable implementation of the **TAGSL structural-learning module integrated with the DGAC clustering pipeline**. A verified configuration on the **Texas** dataset is included as a minimal example for checking that the implementation runs correctly.

## Scope

This release focuses on a minimal runnable example rather than reproducing every experiment reported in the paper. The default Texas configuration is provided to verify the TAGSL structural-learning implementation and its integration with the training pipeline.

The default training pipeline retains the clustering and propagation objectives inherited from DGAC, while replacing the topology-learning branch with TAGSL. Therefore, the result below should be treated as a **reference output for this released configuration**, rather than as a reproduction of every result reported in the paper.

## Environment

A reference Conda environment is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate tagsl
```

The environment was exported from a CUDA/PyTorch setup. PyTorch Geometric extension packages are platform/CUDA dependent, so use versions compatible with your local PyTorch/CUDA installation when necessary.

## Quick Start

From the repository root, simply run:

```bash
python train.py
```

The default run uses:

- dataset: Texas
- 500 training epochs
- five folds/seeds: 0, 1, 2, 3, 4
- TAGSL `dense_exact` backend
- topology-aware affinity
- 100 landmarks

Convenience scripts are also provided.

Linux/macOS:

```bash
bash scripts/run_texas.sh
```

Windows CMD / PowerShell:

```cmd
scripts\run_texas.cmd
```

## Reference Result

A verified five-fold run with the released Texas configuration produced:

| Dataset | ACC | NMI | ARI | F1 |
|---|---:|---:|---:|---:|
| Texas | 71.69 ± 1.36 | 38.73 ± 2.30 | 44.36 ± 4.08 | 42.62 ± 1.22 |

Small numerical differences may occur across hardware/software environments and random-number implementations.

## Data

The minimal release contains the Texas data and the precomputed SVD inputs required by the default run:

```text
dataset/texas/texas_adj.npy
dataset/texas/texas_feat.npy
dataset/texas/texas_label.npy
dataset/pre_saved_U/texas/A_norm.pth
dataset/pre_saved_U/texas/S_norm.pth
```

## Repository Layout

```text
train.py                     training entry point
model.py                     TAGSL model implementation
utils.py                     utility functions
dgc/                         inherited DGAC utilities
dataset/texas/               Texas dataset
dataset/pre_saved_U/texas/   precomputed inputs
scripts/run_texas.sh         Linux/macOS launcher
scripts/run_texas.cmd        Windows launcher
environment.yml              reference environment
```

Training outputs are written automatically to `res_log/` and `res_fold_log/`.

## Acknowledgement

TAGSL is developed on top of the codebase of *Diffusion-based Graph-agnostic Clustering (DGAC, WWW 2025)*. We thank the DGAC authors for releasing their implementation.
# TAGSL
