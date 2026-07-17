import json
import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
from radiomics import featureextractor
from torchradiomics import inject_torch_radiomics
import shutil
from typing import Dict, List, Tuple
import time
import os
import warnings

from data_utils import (whole_panc_lbl, resample_img, GenerateFeatureROI, parse_radiomics_features,  
                        generate_radiomics_config_from_features,create_feature_specific_extractors)

def load_config(config_path):
    """Load radiomics configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def window_image_torch(image, window_min, window_max, output_min=0.0, output_max=400.0, device='cuda', to_int=False):
    """Apply intensity windowing using PyTorch."""
    image_array = sitk.GetArrayFromImage(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    
    if device == 'cuda' and torch.cuda.is_available():
        image_tensor = image_tensor.to(device)
    
    clipped = torch.clamp(image_tensor, window_min, window_max)
    rescaled = output_min + (clipped - window_min) * (output_max - output_min) / (window_max - window_min)
    
    if to_int:
        rescaled = rescaled.round()
    
    if device == 'cuda' and torch.cuda.is_available():
        rescaled_np = rescaled.cpu().numpy()
    else:
        rescaled_np = rescaled.numpy()
    
    if to_int:
        rescaled_np = rescaled_np.astype(np.int32)
    
    result = sitk.GetImageFromArray(rescaled_np)
    result.CopyInformation(image)
    return result


def get_matching_files(image_files: List[Path], mask_files: List[Path]) -> List[Tuple[Path, Path]]:
    """Get matching pairs of image and mask files."""
    
    # Create a mapping of base names to full paths
    image_map = {f.stem.split('.')[0]: f for f in image_files}
    mask_map = {f.stem.split('.')[0]: f for f in mask_files}
    # Find matching pairs
    pairs = []
    for base_name in image_map.keys():
        if base_name in mask_map:
            pairs.append((image_map[base_name], mask_map[base_name]))
    
    return pairs

def create_feature_lookup(features: List[str]) -> Dict[str, int]:
    """Create a lookup table between feature names and indices."""
    return {feature: idx + 1 for idx, feature in enumerate(features)}

def print_mask_info(mask, name="Mask"):
    """Print information about a mask image."""
    mask_array = sitk.GetArrayFromImage(mask)
    unique_values = np.unique(mask_array)
    print(f"\n{name} Information:")
    print(f"  Size: {mask.GetSize()}")
    print(f"  Spacing: {mask.GetSpacing()}")
    print(f"  Origin: {mask.GetOrigin()}")
    print(f"  Unique values: {unique_values}")
    print(f"  Value counts: {dict(zip(*np.unique(mask_array, return_counts=True)))}")

def extract_voxel_radiomics(config_path):
    """Extract voxel-based radiomics features from a directory of images."""
    # Suppress MKL warnings for ill-conditioned matrices
    os.environ['MKL_VERBOSE'] = '0'
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Load configuration
    config = load_config(config_path)
    features = config['features']
    settings = config['settings']
    IO = config['IO']
    IO_opt = IO.keys()
    if 'image_dir' in IO_opt:
        image = sorted(Path(IO['image_dir']).glob('*.nii.gz'))
    else:
        image = [Path(img) for img in IO['image_list']]
    if 'mask_dir' in IO_opt:
        mask = sorted(Path(IO['mask_dir']).glob('*.nii.gz'))
    else:
        mask = [Path(mask) for mask in IO['mask_list']]

    output = IO['output']
    nnunet_format = IO['nnunet_format']
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize torch radiomics with optimized settings
    inject_torch_radiomics()
    
    # Get matching image-mask pairs
    pairs = get_matching_files(image, mask)
    
    if not pairs:
        raise ValueError("No matching image-mask pairs found in the specified directories")
    
    # Create feature lookup table if in nnUNet format
    if nnunet_format:
        feature_lookup = create_feature_lookup(features)
        with open(output_dir / 'feature_lookup.json', 'w') as f:
            json.dump(feature_lookup, f, indent=4)
    
    # Create base extractor with optimized settings
    base_extractor = featureextractor.RadiomicsFeatureExtractor(
        voxelBased=settings['voxelBased'],
        padDistance=settings['padDistance'],
        dtype=torch.float32,
        device=settings['device'],
        binWidth=settings.get('binWidth', 25),  # Default to 25 if not specified
        kernelRadius=settings.get('kernelRadius', 1),  # Default to 1 if not specified
        voxelBatch=settings.get('voxelBatch', 10000)  # Default to 10000 if not specified
    )
    base_extractor.settings['label'] = 1
    
    # Create specialized extractors for individual features
    print("\nCreating specialized extractors for individual features...")
    specialized_extractors = create_feature_specific_extractors(base_extractor, features)
    print(f"Created {len(specialized_extractors)} specialized extractors")
    
    # Verify settings of all extractors
    print("\nVerifying settings of specialized extractors:")
    for feature_name, extractor in specialized_extractors:
        if 'lbp-2d' in feature_name.lower():
            print(f"  {feature_name}: force2D = {extractor.settings.get('force2D', False)}, " +
                  f"image types = {extractor.enabledImagetypes}")
            # Ensure force2D is set to True for LBP-2D features
            extractor.settings['force2D'] = True
    
    # Process each image-mask pair
    for image_path, mask_path in pairs:
        print(f"\nProcessing {image_path.name}...")
        
        try:
            # Load and preprocess images
            basename = image_path.stem.split('.')[0]
            img = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            maskarr = sitk.GetArrayFromImage(mask)
            maskarr[maskarr!=config['settings']['label']] = 0
            maskarr[maskarr==config['settings']['label']] = 1
            mask = sitk.GetImageFromArray(maskarr)
            mask.CopyInformation(img)
            
            # Print initial mask information
            print_mask_info(mask, "Original Mask")
            
            # # Process mask through whole_panc_lbl
            # mask = whole_panc_lbl(mask)
            # print_mask_info(mask, "After whole_panc_lbl")
        
            # Store original image information for later resampling
            ori_img = img
            
            # Resample images
            img = resample_img(img, settings['target_spacing'], is_label=False)
            
            mask = resample_img(mask, settings['target_spacing'], is_label=True)
            print_mask_info(mask, "After resampling")
            
            # Apply windowing
            window_min = settings['window_level'] - settings['window_width'] / 2
            window_max = settings['window_level'] + settings['window_width'] / 2
            img = window_image_torch(img, window_min, window_max, device=settings['device'], to_int=False)
            # Ensure image is float32
            img = sitk.Cast(img, sitk.sitkFloat32)
            
            # Extract features using specialized extractors
            individual_feature_maps = {}
            nnunet_lookup = {}
            
            for feature_name, extractor in specialized_extractors:
                try:
                    print(f"Extracting {feature_name}...")
                    
                    # Print force2D setting if it's an LBP-2D feature
                    if 'lbp-2d' in feature_name.lower():
                        print(f"  force2D setting: {extractor.settings.get('force2D', False)}")
                    
                    # Warn about MCC feature potential issues
                    if '_glcm_MCC' in feature_name or '_glcm_mcc' in feature_name.lower():
                        print(f"  WARNING: MCC feature may encounter ill-conditioned matrices. Errors will be handled gracefully.")
                    
                    extraction_start = time.time()
                    
                    # Extract just this feature
                    result = extractor.execute(img, mask, voxelBased=True)
                    
                    # Find the feature map (non-diagnostic keys)
                    feature_map_key = None
                    for key in result.keys():
                        if 'diagnostics' not in key:
                            feature_map_key = key
                            break
                    
                    if feature_map_key:
                        feature_map = result[feature_map_key]
                        individual_feature_maps[feature_name] = feature_map
                    
                    extraction_end = time.time()
                    print(f"  Extraction time: {extraction_end - extraction_start:.4f} seconds")
                    
                except (RuntimeError, torch._C._LinAlgError) as e:
                    print(f"FAILED: {feature_name} - {str(e)}")
                    print(f"  Skipping this feature due to numerical instability (common with MCC feature)")
                    continue
                except Exception as e:
                    print(f"Error extracting feature {feature_name}: {str(e)}")
                    continue
            
            # Save each individual feature map
            print("\nSaving individual feature maps...")
            for idx, (feature_name, feature_map) in enumerate(individual_feature_maps.items()):
                try:
                    # Process the feature map (apply ROI mask, resample, etc.)
                    feature_array = sitk.GetArrayFromImage(feature_map)
                    mask_array = sitk.GetArrayFromImage(mask)
                    
                    # Apply ROI mask to feature map and enforce float32
                    feature_roi = GenerateFeatureROI(mask_array, feature_array)
                    feature_roi = feature_roi.astype(np.float32, copy=False)
                    print(f"  {feature_name}    feature_roi value range: min={feature_roi.min()}, max={feature_roi.max()}")
                    # Convert back to image
                    feature_image = sitk.GetImageFromArray(feature_roi)
                    feature_image.CopyInformation(img)
                    
                    # Resample to original image space
                    resampled_feature = resample_img(
                        feature_image, 
                        ori_img.GetSpacing(), 
                        is_label=False, 
                        out_origin=ori_img.GetOrigin(), 
                        out_size=ori_img.GetSize(),
                        mask=mask
                    )
                    
                    # Save as NIFTI
                    if nnunet_format:
                        # Create nnUNet format filename - use scan_XXXX format
                        feature_idx = idx + 1

                        output_path = output_dir / f"{basename}_{feature_idx:04d}.nii.gz"
                        nnunet_lookup[feature_name] = feature_idx
                        # Save nnUNet feature lookup dictionary as JSON
                        if feature_idx == len(individual_feature_maps):  # Only save on last feature
                            lookup_path = output_dir / f"feature_lookup.json"
                            with open(lookup_path, 'w') as f:
                                json.dump(nnunet_lookup, f, indent=4)
                            print(f"\nSaved feature lookup mapping to {lookup_path}")
                    else:
                        # Regular format
                        safe_feature_name = feature_name.replace('/', '_').replace(':', '_')
                        output_path = output_dir / f"{basename}_{safe_feature_name}.nii.gz"
                    
                    # Ensure output feature map is float32 before saving
                    resampled_feature = sitk.Cast(resampled_feature, sitk.sitkFloat32)
                    sitk.WriteImage(resampled_feature, str(output_path))
                    print(f"  Saved {feature_name} to {output_path}")
                    
                except Exception as e:
                    print(f"Error saving feature {feature_name}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error processing case {image_path.name}: {str(e)}")
            continue


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract voxel-based radiomics features')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    extract_voxel_radiomics(args.config) 