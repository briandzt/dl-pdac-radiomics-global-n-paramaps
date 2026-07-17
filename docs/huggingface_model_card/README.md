---
license: other
library_name: nnunet
tags:
  - medical-imaging
  - ct
  - radiomics
  - nnunet
  - pancreatic-cancer
  - pdac
  - panorama
pipeline_tag: image-segmentation
---

# Radiomics nnU-Net for PDAC Detection

This model repository contains the nnU-Net checkpoints for the ISBI 2026 work:

**From Global Radiomics to Parametric Maps: A Unified Workflow Fusing Radiomics and Deep Learning for PDAC Detection**

Paper page: <https://huggingface.co/papers/2602.17986>

GitHub code and reproduction instructions: <https://github.com/briandzt/dl-pdac-radiomics-global-n-paramaps>

## Model Description

The workflow combines:

- Stage-1 low-resolution pancreas localization using nnU-Net.
- Voxel radiomics parametric map extraction.
- Global radiomics feature extraction.
- Stage-2 full-resolution PDAC detection using an nnU-Net variant with radiomics channels and global radiomics features.

The public checkpoint layout mirrors the GitHub repository's expected `src/nnUNet_results` structure so users can download the model files directly into a fresh clone.

## Repository Layout

```text
Dataset001_LR/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── plans.json
    ├── fold_0/checkpoint_best.pth
    ├── fold_1/checkpoint_best.pth
    ├── fold_2/checkpoint_best.pth
    ├── fold_3/checkpoint_best.pth
    └── fold_4/checkpoint_best.pth

Dataset002_stage2/
└── nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── plans.json
    ├── fold_0/checkpoint_best.pth
    ├── fold_1/checkpoint_best.pth
    ├── fold_2/checkpoint_best.pth
    ├── fold_3/checkpoint_best.pth
    └── fold_4/checkpoint_best.pth
```

## Intended Use

This release is intended for research reproducibility and method comparison for PDAC detection on venous-phase pancreatic CT.

The workflow outputs:

- a voxel-level PDAC detection map
- a case-level PDAC likelihood JSON file

This model is not intended for clinical deployment or standalone diagnosis.

## How to Use

Clone the GitHub code repository, then download the checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_checkpoints_from_hf.ps1 `
  -RepoId briandzt/radiomics_nnUNet
```

Verify the local setup without a GPU:

```bash
python scripts/verify_repository.py
```

Run local inference:

```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output --image-ext .nii.gz
```

Run Docker inference:

```bash
docker build -t pdac-radiomics-paramaps .
docker run --gpus all --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  pdac-radiomics-paramaps
```

## Inputs and Outputs

Expected input:

- venous-phase pancreatic CT
- NIfTI (`.nii.gz`) for local runs or MHA (`.mha`) for challenge-style Docker input

Expected output:

```text
output/
├── <case>_pdac-likelihood.json
└── images/
    └── pdac-detection-map/
        └── <case>_detection_map.mha
```

## Training Data and Privacy

This model repository does not include patient imaging data. It contains trained model weights and nnU-Net metadata only.

## Limitations

- Requires the GitHub code repository for preprocessing, radiomics extraction, inference, and postprocessing.
- Full inference requires GPU-enabled PyTorch.
- Performance depends on image acquisition, preprocessing compatibility, and domain match to the development data.

## Citation

Please cite the associated arXiv paper if you use this model repository, checkpoints, or reproduction workflow.

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

This workflow also uses [lyhyl/pytorchradiomics](https://github.com/lyhyl/pytorchradiomics) for PyTorch-based radiomics components; please acknowledge that project where appropriate.

## License

License details should be finalized before public release. If the GitHub repository uses a specific license, mirror that license here.
