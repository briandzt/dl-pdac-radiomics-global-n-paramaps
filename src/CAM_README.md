# Class Activation Map (CAM) Generation for PDAC Detection

This module provides tools to generate Class Activation Maps (CAMs) for the nnUNet model with radiomics channels, helping visualize which image regions and channels contribute most to the tumor (label 1) prediction.

## Overview

Two scripts are provided:

1. **`generate_cam.py`** - Full pipeline from raw NIfTI CT image + pancreas mask
2. **`generate_cam_from_prepared.py`** - For pre-prepared nnUNet inputs (9 channels + global features)

## CAM Types Supported

### 1. GradCAM (`--cam_types gradcam`)
Standard Gradient-weighted Class Activation Mapping. Generates a spatial attention map showing which regions of the bottleneck features contribute most to the tumor prediction.

### 2. GradCAM++ (`--cam_types gradcam++`)
Improved version of GradCAM with better localization. Uses weighted positive gradients for more accurate attribution.

### 3. Input Gradient CAM (`--cam_types input_gradient`)
Computes the gradient of the output with respect to each input channel, showing which input channels (CT + 8 radiomics features) contribute most to the tumor prediction. This is most useful for understanding channel importance.

## Input Channels

The model uses 9 input channels:

| Channel | Name |
|---------|------|
| 0 | CT (original image) |
| 1 | feat_log-sigma-4-0-mm-3D_glcm_Correlation |
| 2 | feat_wavelet-LLH_glszm_SizeZoneNonUniformityNormalized |
| 3 | feat_wavelet-HLL_ngtdm_Strength |
| 4 | feat_wavelet-HHL_glcm_Correlation |
| 5 | feat_wavelet-LLL_glcm_Imc1 |
| 6 | feat_wavelet-LLL_glcm_Imc2 |
| 7 | feat_lbp-2D_glrlm_LongRunEmphasis |
| 8 | feat_lbp-2D_glrlm_ShortRunEmphasis |

## Usage

### Option 1: From Raw CT Image (Full Pipeline)

If you have a raw CT image and pancreas segmentation mask:

```bash
python generate_cam.py \
    --input_image /path/to/ct_image.nii.gz \
    --pancreas_mask /path/to/pancreas_mask.nii.gz \
    --output_dir /path/to/output \
    --cam_types gradcam++ input_gradient \
    --target_class 1 \
    --device cuda
```

**Arguments:**
- `--input_image, -i`: Path to input CT image (NIfTI format)
- `--pancreas_mask, -m`: Path to pancreas segmentation mask
- `--output_dir, -o`: Output directory for CAM files
- `--model_dir`: Path to trained model (default: `./nnUNet_results/Dataset002_stage2`)
- `--voxel_config`: Path to voxel radiomics config (default: `./PANORAMA_voxel.json`)
- `--cam_types`: CAM types to generate (`gradcam`, `gradcam++`, `input_gradient`)
- `--target_class`: Target class index (1 for tumor)
- `--folds`: Model folds to use (default: 0)
- `--checkpoint`: Checkpoint filename (default: `checkpoint_best.pth`)
- `--device`: cuda or cpu

### Option 2: From Pre-Prepared Inputs

If you already have prepared nnUNet inputs (9 channels + global features):

```bash
# Single case
python generate_cam_from_prepared.py \
    --input_dir /path/to/prepared_inputs \
    --casename PANORAMA_0001 \
    --output_dir /path/to/output \
    --cam_types gradcam++ input_gradient

# Batch processing (all cases in directory)
python generate_cam_from_prepared.py \
    --input_dir /path/to/prepared_inputs \
    --output_dir /path/to/output \
    --cam_types gradcam++ input_gradient
```

**Expected input file structure:**
```
input_dir/
├── CASENAME_0000.nii.gz  (CT image)
├── CASENAME_0001.nii.gz  (radiomics feature 1)
├── CASENAME_0002.nii.gz  (radiomics feature 2)
├── ...
├── CASENAME_0008.nii.gz  (radiomics feature 8)
└── CASENAME_global.npy   (global radiomics features)
```

## Output Files

The scripts generate the following outputs:

### Spatial CAMs (GradCAM/GradCAM++)
- `{casename}_gradcam++_spatial.nii.gz` - Overall spatial attention map

### Per-Channel CAMs (Input Gradient)
- `{casename}_input_grad_ch0_CT.nii.gz` - CAM for CT channel
- `{casename}_input_grad_ch1_{feature_name}.nii.gz` - CAM for radiomics channel 1
- ... (one file per channel)
- `{casename}_input_grad_combined.nii.gz` - Combined channel CAM

### Summary
- `{casename}_channel_importance.json` - JSON file with mean CAM values per channel (channel importance ranking)

## Example: Channel Importance Analysis

The `channel_importance.json` file contains the average CAM value for each channel, useful for understanding which features contribute most to tumor detection:

```json
{
    "CT": 0.234,
    "feat_wavelet-HHL_glcm_Correlation": 0.189,
    "feat_lbp-2D_glrlm_ShortRunEmphasis": 0.156,
    ...
}
```

Higher values indicate greater contribution to the tumor prediction.

## Visualization

The output NIfTI files can be visualized in any medical imaging software:
- **3D Slicer**
- **ITK-SNAP**
- **FSLeyes**

To overlay CAM on the original image:
1. Load the original CT image
2. Load the CAM file as an overlay
3. Apply a color map (e.g., "hot" or "jet")
4. Adjust opacity to see through to the original image

## Requirements

- Python 3.8+
- PyTorch
- SimpleITK
- NumPy
- PyRadiomics (modified version in packages/)
- torchradiomics (GPU-accelerated radiomics)

## Notes

1. **GPU Memory**: GradCAM computation requires additional GPU memory. For large volumes, you may need to use `--device cpu` or reduce the input size.

2. **Target Class**: The default target class is 1 (tumor). Other available classes:
   - 0: Background
   - 1: Tumor
   - 2: Veins
   - 3: Arteries
   - 4: Pancreas
   - 5: Pancreatic duct
   - 6: Common bile duct

3. **Preprocessing**: The full pipeline (`generate_cam.py`) uses the same preprocessing as `process.py` / `process_local.py`, including:
   - Pancreas ROI cropping
   - Resampling
   - CT windowing
   - Voxel radiomics extraction
   - Global radiomics extraction

4. **Model Attention**: The model uses a CrossAttentionBlock that fuses global radiomics features with bottleneck spatial features. This attention mechanism is captured in the GradCAM computation.

