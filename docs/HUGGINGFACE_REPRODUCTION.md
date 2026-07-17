# Hugging Face Model Release and Reproduction Tutorial

This guide keeps the reproducible code on GitHub and the large nnU-Net checkpoint weights on Hugging Face Hub.

Paper page:

- <https://huggingface.co/papers/2602.17986>

Model repository:

- <https://huggingface.co/briandzt/radiomics_nnUNet>

## Recommended Split Between GitHub and Hugging Face

Use GitHub for:

- Dockerfile and Python inference code
- nnU-Net metadata files: `dataset.json`, `dataset_fingerprint.json`, `plans.json`
- configuration files and helper scripts
- documentation and citations

Use Hugging Face for:

- Stage-1 pancreas segmentation checkpoints
- Stage-2 radiomics/global/parametric-map nnU-Net checkpoints
- model card and paper links
- release tags for paper/revision versions

Do not upload patient images, private data, generated challenge outputs, or local cache folders.

## One-Time CLI Setup

Install or activate the Hugging Face CLI:

```powershell
conda activate hf-cli
hf --version
hf auth login
```

On this workstation, the working CLI is:

```powershell
C:\Users\yiyh1\anaconda3\envs\hf-cli\Scripts\hf.exe
```

## Recommended Hugging Face Repository Layout

Upload files to the Hugging Face model repo using the same relative layout expected by this GitHub repository:

```text
briandzt/radiomics_nnUNet/
├── README.md
├── Dataset001_LR/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       ├── dataset.json
│       ├── dataset_fingerprint.json
│       ├── plans.json
│       ├── fold_0/checkpoint_best.pth
│       ├── fold_1/checkpoint_best.pth
│       ├── fold_2/checkpoint_best.pth
│       ├── fold_3/checkpoint_best.pth
│       └── fold_4/checkpoint_best.pth
└── Dataset002_stage2/
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

The Stage-2 source files may be named `checkpoint_1009.pth` locally. For the public model repo, upload them as `checkpoint_best.pth` so the inference code can use the standard nnU-Net checkpoint name.

## Upload the Model to Hugging Face

From the GitHub repository root:

```powershell
cd C:\Users\yiyh1\Documents\Winter2026\PANORAMA\dl-pdac-radiomics-global-n-paramaps
conda activate hf-cli
```

Confirm checkpoint readiness:

```powershell
python scripts\verify_repository.py
```

Upload the model card:

```powershell
hf upload briandzt/radiomics_nnUNet .\docs\huggingface_model_card\README.md README.md `
  --repo-type model `
  --commit-message "Add PANORAMA radiomics nnUNet model card"
```

Upload the model tree:

```powershell
hf upload briandzt/radiomics_nnUNet .\src\nnUNet_results . `
  --repo-type model `
  --include "Dataset001_LR/**" `
  --include "Dataset002_stage2/**" `
  --commit-message "Add PANORAMA radiomics nnUNet checkpoints"
```

If you prefer to stage a pull request on Hugging Face first:

```powershell
hf upload briandzt/radiomics_nnUNet .\src\nnUNet_results . `
  --repo-type model `
  --include "Dataset001_LR/**" `
  --include "Dataset002_stage2/**" `
  --create-pr `
  --commit-message "Add PANORAMA radiomics nnUNet checkpoints"
```

## Suggested Hugging Face Model Card Content

I drafted a model card at:

```text
docs/huggingface_model_card/README.md
```

Upload it to `briandzt/radiomics_nnUNet/README.md` after finalizing the license.

The model card at `briandzt/radiomics_nnUNet/README.md` should include:

- model name and short description
- link to the paper page: <https://huggingface.co/papers/2602.17986>
- link to the GitHub code repository
- intended use: PDAC detection research on venous-phase pancreatic CT
- input requirements and expected image format
- output description: voxel-level PDAC detection map and case-level likelihood JSON
- checkpoint layout and fold ensemble details
- citation
- license and usage restrictions
- data/privacy note explaining that no patient data are included

## Fresh Reproduction on a New Computer

Clone the GitHub repository:

```bash
git clone https://github.com/briandzt/dl-pdac-radiomics-global-n-paramaps.git
cd dl-pdac-radiomics-global-n-paramaps
```

Create the environment:

```bash
conda create -n pdac-rad python=3.10 -y
conda activate pdac-rad
pip install -r requirements.txt
pip install -e src/dynamic-network-architectures_global_rad/dynamic-network-architectures
pip install -e src/nnunetv2_global_rad
pip install -e src/pyradiomics-3.1.0-Zengtian
pip install -e src/pytorchradiomics-main
pip install -e src/report-guided-annotation
```

Download checkpoints from Hugging Face:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_checkpoints_from_hf.ps1 `
  -RepoId briandzt/radiomics_nnUNet
```

Verify the repository and checkpoints without a GPU:

```bash
python scripts/verify_repository.py
```

Run local inference:

```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output --image-ext .nii.gz
```

For challenge-style `.mha` inputs:

```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output --image-ext .mha
```

## Docker Reproduction

After checkpoints have been downloaded into `src/nnUNet_results`, build the Docker image:

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

Challenge-style input directory:

```text
/input/images/venous-ct/
```

## Expected Outputs

```text
output/
├── <case>_pdac-likelihood.json
└── images/
    └── pdac-detection-map/
        └── <case>_detection_map.mha
```

## Practical Release Advice

1. Upload the Hugging Face model repo first and confirm all 10 checkpoints are present.
2. Run `scripts/verify_repository.py` after downloading from Hugging Face into a clean clone.
3. Add the Hugging Face model URL to the GitHub README.
4. Add the GitHub repo URL and paper URL to the Hugging Face model card.
5. Use a release tag on GitHub and a matching tag or commit note on Hugging Face for the exact paper version.
