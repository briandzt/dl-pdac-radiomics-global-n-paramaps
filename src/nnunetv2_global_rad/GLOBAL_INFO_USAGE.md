# Using Global Information with nnUNetv2_predict

This modified version of nnUNet supports passing global information (radiomics features, clinical data, etc.) to the network during prediction.

## Requirements

- Global info files should be stored as `.npy` files
- Naming convention: `{case_name}_global.npy` (e.g., `102223_00001_global.npy`)
- Each file should contain a numpy array of shape (1, num_features)
- Place all global info files in a single directory

## Command Line Usage

### Option 1: Using nnUNetv2_predict directly

```bash
nnUNetv2_predict \
    -d Dataset002_stage2 \
    -i /path/to/input/images \
    -o /path/to/output \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth \
    -global_info_folder /path/to/global_info/folder
```

### Option 2: Using Python API

```python
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Initialize predictor with global_info_folder
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=False,
    allow_tqdm=True,
    global_info_folder='/path/to/nnUNet_preprocessed/Dataset002_stage2/nnUNetPlans_3d_fullres'
)

# Load trained model
predictor.initialize_from_trained_model_folder(
    model_folder='/path/to/nnUNet_results/Dataset002_stage2/nnUNetTrainer__nnUNetPlans__3d_fullres',
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name='checkpoint_final.pth'
)

# Run prediction
predictor.predict_from_files(
    list_of_lists_or_source_folder='/path/to/input/images',
    output_folder_or_list_of_truncated_output_files='/path/to/output',
    save_probabilities=True,
    overwrite=True,
    num_processes_preprocessing=3,
    num_processes_segmentation_export=3
)
```

## Example for PANORAMA Dataset

For the Dataset002_stage2 example in this repository:

```bash
# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /common/dengz2/nnunet_globalrad

# Set environment variables
export nnUNet_raw="/common/lidxxlab/Zengtian/PANORAMA/nnUNet_structures/nnUNet_raw"
export nnUNet_preprocessed="/common/lidxxlab/Zengtian/PANORAMA/nnUNet_structures/nnUNet_preprocessed"
export nnUNet_results="/common/lidxxlab/Zengtian/PANORAMA/nnUNet_structures/nnUNet_results"

# Run prediction with global info
nnUNetv2_predict \
    -d Dataset002_stage2 \
    -i $nnUNet_raw/Dataset002_stage2/validation_data0 \
    -o $nnUNet_raw/Dataset002_stage2/validation_tmp0 \
    -c 3d_fullres \
    -tr nnUNetTrainer_Loss_CE_checkpoints \
    -f 0 \
    -chk checkpoint_final.pth \
    -global_info_folder $nnUNet_preprocessed/Dataset002_stage2/nnUNetPlans_3d_fullres \
    --save_probabilities
```

## Notes

- If `-global_info_folder` is not specified, prediction will work without global info (if the network supports it)
- The global info file for each case is matched by extracting the case name from the input filename
- Global info is automatically loaded and passed to the network during sliding window prediction
- The same global info vector is used for all patches of the same image

