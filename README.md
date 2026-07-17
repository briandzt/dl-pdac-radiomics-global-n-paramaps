# From Global Radiomics to Parametric Maps for PDAC Detection

[ISBI 2026] Reproducible inference code for **From Global Radiomics to Parametric Maps: A Unified Workflow Fusing Radiomics and Deep Learning for PDAC Detection**.

This repository is organized in the same spirit as PANORAMA solution repositories such as [han-liu/PanDx](https://github.com/han-liu/PanDx): a top-level local entrypoint, vendored packages, an nnU-Net workspace, model-download instructions, and clear inference/output sections. The Docker workflow remains the canonical reproduction path for the accepted grand-challenge submission.

Paper page: <https://huggingface.co/papers/2602.17986>

Model weights: <https://huggingface.co/briandzt/radiomics_nnUNet>

Hugging Face upload/download tutorial: [docs/HUGGINGFACE_REPRODUCTION.md](docs/HUGGINGFACE_REPRODUCTION.md)

## Repository Layout

```text
dl-pdac-radiomics-global-n-paramaps/
├── Dockerfile
├── main.py
├── requirements.txt
├── scripts/
│   ├── prepare_checkpoints.ps1
│   ├── download_checkpoints_from_hf.ps1
│   └── verify_repository.py
├── src/
│   ├── process.py
│   ├── process_local.py
│   ├── data_utils.py
│   ├── voxel_radiomics.py
│   ├── nnUNet_results/
│   ├── nnunetv2_global_rad/
│   ├── dynamic-network-architectures_global_rad/
│   ├── pyradiomics-3.1.0-Zengtian/
│   ├── pytorchradiomics-main/
│   └── report-guided-annotation/
└── workspace/                  # create locally, ignored by git
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
    └── test_example/
        ├── input/
        └── output/
```

`src/nnUNet_results/` contains model metadata and fold folders. Checkpoint weights are intentionally not tracked by git.

## Installation

### Requirements

- Linux or Windows with PowerShell for the helper scripts
- NVIDIA GPU for full inference
- CUDA-compatible PyTorch
- Python 3.10 recommended for parity with the Dockerfile

### Create an environment

```bash
conda create -n pdac-rad python=3.10 -y
conda activate pdac-rad
pip install -r requirements.txt
```

Install the vendored packages in editable mode:

```bash
pip install -e src/dynamic-network-architectures_global_rad/dynamic-network-architectures
pip install -e src/nnunetv2_global_rad
pip install -e src/pyradiomics-3.1.0-Zengtian
pip install -e src/pytorchradiomics-main
pip install -e src/report-guided-annotation
```

## Model Checkpoints

The Python workflow uses `checkpoint_best.pth` for every fold. The full-resolution checkpoints downloaded as `checkpoint_1009.pth` should be placed/linked under the repo as `checkpoint_best.pth`.

Required model layout:

```text
src/nnUNet_results/
├── Dataset001_LR/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       ├── fold_0/checkpoint_best.pth
│       ├── fold_1/checkpoint_best.pth
│       ├── fold_2/checkpoint_best.pth
│       ├── fold_3/checkpoint_best.pth
│       └── fold_4/checkpoint_best.pth
└── Dataset002_stage2/
    └── nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/
        ├── fold_0/checkpoint_best.pth
        ├── fold_1/checkpoint_best.pth
        ├── fold_2/checkpoint_best.pth
        ├── fold_3/checkpoint_best.pth
        └── fold_4/checkpoint_best.pth
```

If your Stage-2 downloads are stored as:

```text
PANORAMA/
├── fold0/checkpoint_1009.pth
├── fold1/checkpoint_1009.pth
├── fold2/checkpoint_1009.pth
├── fold3/checkpoint_1009.pth
└── fold4/checkpoint_1009.pth
```

prepare local hard links with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_checkpoints.ps1 `
  -CheckpointRoot C:\Users\yiyh1\Documents\Winter2026\PANORAMA `
  -Stage1ModelDir C:\Users\yiyh1\Documents\Winter2026\PANORAMA\submission_fullrad\PANORAMA_submission_voxrad\src\nnUNet_results\Dataset001_LR\nnUNetTrainer__nnUNetPlans__3d_fullres `
  -Link
```

Omit `-Link` to copy instead of hard-linking. Do not commit `.pth` files.

The model weights are hosted on Hugging Face. Fresh users can download the checkpoints with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_checkpoints_from_hf.ps1 `
  -RepoId briandzt/radiomics_nnUNet
```

See [docs/HUGGINGFACE_REPRODUCTION.md](docs/HUGGINGFACE_REPRODUCTION.md) for the full GitHub + Hugging Face release workflow.

## CPU-Safe Verification

This does not run inference and does not require a GPU:

```bash
python scripts/verify_repository.py
```

It checks required files, nnU-Net metadata, fold folders, and whether both Stage-1 and Stage-2 checkpoints are locally available.

## Local Inference

Set nnU-Net environment variables if desired:

```bash
export nnUNet_raw="./workspace/nnUNet_raw"
export nnUNet_preprocessed="./workspace/nnUNet_preprocessed"
export nnUNet_results="./src/nnUNet_results"
```

Run:

```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output --image-ext .nii.gz
```

For challenge-style `.mha` inputs:

```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output --image-ext .mha
```

The workflow uses all five folds: `0,1,2,3,4`.

## Docker Inference

Build:

```bash
docker build -t pdac-radiomics-paramaps .
```

Run:

```bash
docker run --gpus all --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  pdac-radiomics-paramaps
```

Challenge-style input path:

```text
/input/images/venous-ct/
```

## Outputs

Outputs are written in challenge-compatible form:

```text
output/
├── <case>_pdac-likelihood.json
└── images/
    └── pdac-detection-map/
        └── <case>_detection_map.mha
```

The detection map stores voxel-level PDAC likelihood. The JSON stores the case-level PDAC likelihood score.

## Notes

- Checkpoints are excluded from git via `.gitignore`.
- `checkpoint_1009.pth` is used locally after being renamed or linked to `checkpoint_best.pth`.
- `process.py` is the canonical workflow; `main.py` and `src/process_local.py` are local convenience wrappers.
- Full inference requires GPU-enabled PyTorch and nnU-Net dependencies.

## Citation

If you use this repository, model weights, or reproduction workflow, please cite the associated arXiv paper. A BibTeX copy is also available in [CITATION.bib](CITATION.bib).

```bibtex
@misc{deng2026globalradiomicsparametricmaps,
  title         = {From Global Radiomics to Parametric Maps: A Unified Workflow Fusing Radiomics and Deep Learning for PDAC Detection},
  author        = {Deng, Zengtian and He, Yimeng and Shi, Yu and Wang, Lixia and Qureshi, Touseef Ahmad and Huang, Xiuzhen and Li, Debiao},
  year          = {2026},
  eprint        = {2602.17986},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  doi           = {10.48550/arXiv.2602.17986},
  url           = {https://arxiv.org/abs/2602.17986}
}
```

## Acknowledgement

This workflow builds on nnU-Net, PyRadiomics, [lyhyl/pytorchradiomics](https://github.com/lyhyl/pytorchradiomics), and report-guided annotation utilities. Please cite or acknowledge the original tools where appropriate.
