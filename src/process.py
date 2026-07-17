
#  Copyright 2024 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#  Copyright 2024 Zengtian Deng, Cedars Sinai,Los Angeles, California, USA
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Modifications:
#  - Add radiomics extraction and modified pipeline


import os
import glob
import argparse
import SimpleITK as sitk
import numpy as np
import time
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# imports required for running nnUNet algorithm
import subprocess
from subprocess import check_output, STDOUT, CalledProcessError
from pathlib import Path
import json
# imports required for my algorithm
from data_utils import whole_panc_lbl,resample_img, CropPancreasROI, GetFullSizDetectionMap,GetFullSizDetectionMap_nifti,PostProcessing,GenerateFeatureROI
import radiomics
from radiomics import featureextractor 
from scipy.ndimage import binary_dilation
import shutil
import torch
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")
import tempfile

def extract_voxel_radiomics_features(image, mask, output_dir, config_path,basename):
    """
    Extract voxel-based radiomics features using the voxel_radiomics.py approach.
    
    Args:
        image: SimpleITK image
        mask: SimpleITK mask
        output_dir: Path to output directory
        config_path: Path to configuration JSON file
    
    Returns:
        List of feature map paths in nnUNet format
    """
    # Create temporary directories for input
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_dir = temp_path / "images"
        mask_dir = temp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()
        
        # Save image and mask to temporary directories
        image_path = image_dir / f"{basename}.nii.gz"
        mask_path = mask_dir / f"{basename}.nii.gz"
        sitk.WriteImage(image, str(image_path))
        sitk.WriteImage(mask, str(mask_path))
        
        # Load configuration and modify for this specific case
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update IO settings for temporary files
        config['IO'] = {
            'image_dir': str(image_dir),
            'mask_dir': str(mask_dir),
            'output': str(output_dir),
            'nnunet_format': True
        }
        
        # Save modified config
        temp_config_path = temp_path / "temp_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Import and run voxel_radiomics extraction
        from voxel_radiomics import extract_voxel_radiomics
        extract_voxel_radiomics(str(temp_config_path))
        
        # Return list of generated feature maps
        feature_maps = []
        for i in range(1, len(config['features']) + 1):
            feature_path = output_dir / f"{basename}_{i:04d}.nii.gz"
            if feature_path.exists():
                feature_maps.append(feature_path)
        
        return feature_maps

def extract_global_radiomics_features(image, mask, output_dir, config_path):
    """
    Extract global-based radiomics features using the global_radiomics.py approach.
    """
    from data_utils import create_feature_specific_extractors
    from voxel_radiomics import window_image_torch,load_config
    features = np.array(['feat_log-sigma-3-0-mm-3D_glcm_Correlation',
                     'feat_wavelet-LLH_glcm_Imc1',
                     'feat_wavelet-HLL_glszm_SmallAreaEmphasis',
                     'feat_wavelet-HHL_firstorder_Mean',
                     'feat_wavelet-HHL_glcm_Correlation', 
                     'feat_wavelet-LLL_glcm_MCC',
                     'feat_gradient_glcm_Imc1', 
                     'feat_lbp-2D_glrlm_ShortRunEmphasis',
                     'feat_original_shape_Sphericity',
                     'feat_original_shape_SurfaceVolumeRatio'])
    config = load_config(config_path)
    settings = config['settings']
    img = resample_img(image, settings['target_spacing'], is_label=False)
    mask = resample_img(mask, settings['target_spacing'], is_label=True)
    window_min = settings['window_level'] - settings['window_width'] / 2
    window_max = settings['window_level'] + settings['window_width'] / 2
    img = window_image_torch(img, window_min, window_max, device=settings['device'], to_int=True)
    base_extractor = featureextractor.RadiomicsFeatureExtractor(
        voxelBased=False,
        padDistance=10,
        dtype=torch.float32,
        device='cpu',
        binWidth=10
    )
    base_extractor.settings['label'] = 1
    specialized_extractors = create_feature_specific_extractors(base_extractor, features)
    global_features = []
    for feature_name, extractor in specialized_extractors:
        print(f"Extracting {feature_name}...")
        # Print force2D setting if it's an LBP-2D feature
        if 'lbp-2d' in feature_name.lower():
            print(f"  force2D setting: {extractor.settings.get('force2D', False)}")
        extraction_start = time.time()
        # Extract just this feature
        result = extractor.execute(img, mask, voxelBased=False)

        # Find the feature map (non-diagnostic keys)
        feature_map_key = None
        for key in result.keys():
            if 'diagnostics' not in key:
                feature_map_key = key
                break
        if not feature_map_key:
            raise ValueError(f"No valid feature map key found in results")
        feature_value = result[feature_map_key]
        global_features.append(float(feature_value))
    global_features = np.array(global_features)
    global_features = global_features.reshape(1, -1)
    return global_features
def _prepare_fullres_case(args):
    casename, image_path, nnunet_input_dir_lowres, nnunet_output_dir_lowres, nnunet_input_dir_fullres, output_dir_images, voxel_config_path = args
    # read original image
    itk_img = sitk.ReadImage(image_path, sitk.sitkFloat32)
    # load lowres prediction
    mask_pred_path = str(Path(nnunet_output_dir_lowres) / f"{casename}.nii.gz")
    mask_low_res_base = sitk.ReadImage(mask_pred_path)
    mask_low_res_base = whole_panc_lbl(mask_low_res_base)
    # Verify mask has non-zero values
    mask_array = sitk.GetArrayFromImage(mask_low_res_base)
    if np.sum(mask_array) == 0:
        raise ValueError(f"Mask {casename} contains no non-zero values after whole_panc_lbl processing")
    mask_low_res_arr = sitk.GetArrayFromImage(mask_low_res_base)
    dilated_pancreas_mask = binary_dilation(mask_low_res_arr, structure=np.ones((3,3,3), dtype=bool))
    mask_low_res = sitk.GetImageFromArray(dilated_pancreas_mask.astype(int))
    mask_low_res.CopyInformation(mask_low_res_base)

    crop_margins = [100,50,15]
    cropped_image, crop_coordinates,_ = CropPancreasROI(itk_img, mask_low_res, crop_margins)
    cropped_label,crop_coordinateslb,_ = CropPancreasROI(mask_low_res, mask_low_res, crop_margins)
    cropped_label = resample_img(cropped_label,
                                 cropped_image.GetSpacing(),
                                 is_label=True,
                                 out_origin=cropped_image.GetOrigin(),
                                 out_size=cropped_image.GetSize())
    sitk.WriteImage(cropped_image, str(Path(nnunet_input_dir_fullres) / f"{casename}_0000.nii.gz"))

    # voxel radiomics features
    samp_im = resample_img(cropped_image,[1.0,1.0,2.5], is_label=False,out_origin=cropped_image.GetOrigin())
    samp_msk = resample_img(cropped_label,samp_im.GetSpacing(), is_label=True, out_origin=samp_im.GetOrigin(),out_size=samp_im.GetSize())

    temp_feature_dir = Path(nnunet_input_dir_fullres) / f"temp_features_{casename}"
    temp_feature_dir.mkdir(exist_ok=True)

    feature_maps = extract_voxel_radiomics_features(samp_im, samp_msk, temp_feature_dir, str(voxel_config_path),casename)
    global_features = extract_global_radiomics_features(samp_im, samp_msk, temp_feature_dir, str(voxel_config_path))
    global_features_path = Path(nnunet_input_dir_fullres) / f"{casename}_global.npy"
    np.save(str(global_features_path), global_features)

    samp_imnew = resample_img(samp_im,cropped_image.GetSpacing(), is_label=False, out_origin=cropped_image.GetOrigin(),out_size=cropped_image.GetSize())
    for i, feature_path in enumerate(feature_maps):
        if feature_path.exists():
            feat_map = sitk.ReadImage(str(feature_path))
            feat_map_resampled = resample_img(
                feat_map,
                samp_imnew.GetSpacing(),
                is_label=False,
                out_origin=samp_imnew.GetOrigin(),
                out_size=samp_imnew.GetSize()
            )
            output_path = Path(nnunet_input_dir_fullres) / f"{casename}_{i+1:04d}.nii.gz"
            sitk.WriteImage(feat_map_resampled, str(output_path))

    shutil.rmtree(temp_feature_dir, ignore_errors=True)
    return casename, crop_coordinates, str(Path(nnunet_input_dir_fullres) / f"{casename}.npz"), str(Path(nnunet_input_dir_fullres) / f"{casename}.nii.gz")

class PDACDetectionContainer(SegmentationAlgorithm):
    def __init__(
        self,
        nnunet_base=None,
        input_dir=None,
        output_dir=None,
        model_dir=None,
        image_ext=".mha",
    ):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        nnunet_base = Path(nnunet_base or os.environ.get("PANORAMA_NNUNET_BASE", "/opt/algorithm/nnunet"))
        self.nnunet_input_dir_lowres = nnunet_base / "input_lowres"
        self.nnunet_input_dir_fullres = nnunet_base / "input_fullres"
        self.nnunet_output_dir_lowres = nnunet_base / "output_lowres"
        self.nnunet_output_dir_fullres = nnunet_base / "output_fullres"
        self.nnunet_model_dir = Path(
            model_dir
            or os.environ.get("nnUNet_results")
            or os.environ.get("PANORAMA_MODEL_DIR")
            or (nnunet_base / "nnUNet_results")
        )
       
        # input / output paths
        self.ct_ip_dir = Path(input_dir or os.environ.get("PANORAMA_INPUT_DIR", "/input/images/venous-ct"))
        self.clinical_info_path = os.environ.get(
            "PANORAMA_CLINICAL_INFO",
            "/input/clinical-information-pancreatic-ct.json",
        )
        self.output_dir = Path(output_dir or os.environ.get("PANORAMA_OUTPUT_DIR", "/output"))
        self.image_ext = image_ext

        self.output_dir_images  = Path(os.path.join(self.output_dir,"images")) 
        self.output_dir_tlm     = Path(os.path.join(self.output_dir_images,"pdac-detection-map")) 
        self.detection_map      = self.output_dir_tlm / "detection_map.mha"

        # ensure required folders exist
        self.nnunet_input_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_input_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_tlm.mkdir(exist_ok=True, parents=True)
        


        self.voxel_config_path = Path(__file__).resolve().parent / "PANORAMA_voxel.json"
        if not self.voxel_config_path.exists():
            raise FileNotFoundError(f"Voxel radiomics config not found at {self.voxel_config_path}. Update self.voxel_config_path.")
        
        self.ct_images = sorted(glob.glob(os.path.join(self.ct_ip_dir, f"*{self.image_ext}")))
        print(self.ct_images)
        if not self.ct_images:
            print(f"No {self.image_ext} images found in input directory: {self.ct_ip_dir}")

    def process(self):
        """
        Load CT scan and Generate Heatmap for Pancreas Cancer  
        """
        if not self.ct_images:
            print('No input images found')
            return

        for idx, image_path in enumerate(self.ct_images):
            print('processing ', idx, image_path)
            itk_img = sitk.ReadImage(image_path, sitk.sitkFloat32)

            # Get low resolution pancreas segmentation.
            basename = os.path.basename(image_path)
            basename = basename.replace(self.image_ext, ".nii.gz")
            casename = basename.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            new_spacing = (4.5, 4.5, 9.0)
            image_resampled = resample_img(itk_img, new_spacing, is_label=False, out_origin=itk_img.GetOrigin(), out_size=[])
            sitk.WriteImage(image_resampled, str(self.nnunet_input_dir_lowres / basename))
            print('predict LR')
            self.predict(
                input_dir=self.nnunet_input_dir_lowres,
                output_dir=self.nnunet_output_dir_lowres,
                task="001",
                checkpoint='checkpoint_best.pth',
                store_probability_maps=False
            )
            print('predict LR done')
            mask_pred_path = str(self.nnunet_output_dir_lowres / f"{casename}.nii.gz")
            mask_low_res_base = sitk.ReadImage(mask_pred_path)
            mask_low_res_base = whole_panc_lbl(mask_low_res_base)
            mask_low_res_arr = sitk.GetArrayFromImage(mask_low_res_base)
            dilated_pancreas_mask = binary_dilation(mask_low_res_arr, structure=np.ones((3, 3, 3), dtype=bool))
            mask_low_res = sitk.GetImageFromArray(dilated_pancreas_mask.astype(int))
            mask_low_res.CopyInformation(mask_low_res_base)

            crop_margins = [100, 50, 15]
            cropped_image, crop_coordinates, _ = CropPancreasROI(itk_img, mask_low_res, crop_margins)
            cropped_label, crop_coordinateslb, _ = CropPancreasROI(mask_low_res, mask_low_res, crop_margins)
            cropped_label = resample_img(
                cropped_label,
                cropped_image.GetSpacing(),
                is_label=True,
                out_origin=cropped_image.GetOrigin(),
                out_size=cropped_image.GetSize()
            )
            sitk.WriteImage(cropped_image, str(self.nnunet_input_dir_fullres / f"{casename}_0000.nii.gz"))

            # Use predicted pancreas to extract voxel radiomic feature maps.
            samp_im = resample_img(cropped_image, [1.0, 1.0, 2.5], is_label=False, out_origin=cropped_image.GetOrigin())
            samp_msk = resample_img(
                cropped_label,
                samp_im.GetSpacing(),
                is_label=True,
                out_origin=samp_im.GetOrigin(),
                out_size=samp_im.GetSize(),
            )

            temp_feature_dir = self.nnunet_input_dir_fullres / f"temp_features_{casename}"
            temp_feature_dir.mkdir(exist_ok=True)

            feature_maps = extract_voxel_radiomics_features(samp_im, samp_msk, temp_feature_dir, str(self.voxel_config_path), casename)
            global_features = extract_global_radiomics_features(samp_im, samp_msk, temp_feature_dir, str(self.voxel_config_path))
            global_features_path = self.nnunet_input_dir_fullres / f"{casename}_global.npy"
            np.save(str(global_features_path), global_features)

            # Resample back to cropped image space and save as nnUNet channels.
            samp_imnew = resample_img(samp_im, cropped_image.GetSpacing(), is_label=False, out_origin=cropped_image.GetOrigin(), out_size=cropped_image.GetSize())
            for i, feature_path in enumerate(feature_maps):
                if feature_path.exists():
                    feat_map = sitk.ReadImage(str(feature_path))
                    feat_map_resampled = resample_img(
                        feat_map,
                        samp_imnew.GetSpacing(),
                        is_label=False,
                        out_origin=samp_imnew.GetOrigin(),
                        out_size=samp_imnew.GetSize()
                    )
                    output_path = self.nnunet_input_dir_fullres / f"{casename}_{i+1:04d}.nii.gz"
                    sitk.WriteImage(feat_map_resampled, str(output_path))

            shutil.rmtree(temp_feature_dir, ignore_errors=True)
            print('predicting map')
            self.predict(
                input_dir=self.nnunet_input_dir_fullres,
                output_dir=self.nnunet_output_dir_fullres,
                task="002",
                trainer="nnUNetTrainer_Loss_CE_checkpoints",
                checkpoint='checkpoint_best.pth',
                folds="0,1,2,3,4",
                store_probability_maps=True,
                global_info_folder=self.nnunet_input_dir_fullres
            )
            print('predicting map done')

            # Postprocess and map back to full image space.
            pred_path_npz = str(self.nnunet_output_dir_fullres / f"{casename}.npz")
            prediction = np.load(pred_path_npz)
            pred_path_nifti = str(self.nnunet_output_dir_fullres / f"{casename}.nii.gz")
            prediction_postprocessed = PostProcessing(prediction, pred_path_nifti)
            detection_map, patient_level_prediction = GetFullSizDetectionMap(prediction_postprocessed, crop_coordinates, itk_img)

            case_detection_map = self.output_dir_tlm / f"{casename}_detection_map.mha"
            sitk.WriteImage(detection_map, case_detection_map)
            write_json_file(location=self.output_dir / f"{casename}_pdac-likelihood.json", content=patient_level_prediction)




    def predict(self, input_dir, output_dir, task="Task103_AllStructures", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2,3,4", 
                    store_probability_maps=True,global_info_folder=None,
                    num_processes_preprocessing=None, num_processes_segmentation_export=None):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                cmd.extend(folds.split(','))

            if checkpoint:
                cmd.append('-chk')
                cmd.append(checkpoint)

            if store_probability_maps:
                cmd.append('--save_probabilities')
            
            if global_info_folder:
                cmd.append('-global_info_folder')
                cmd.append(str(global_info_folder))

            # Optional process controls (only append if nnUNetv2 CLI supports them; safe to ignore otherwise)
            if num_processes_preprocessing is not None:
                cmd.append('--num_processes_preprocessing')
                cmd.append(str(num_processes_preprocessing))
            if num_processes_segmentation_export is not None:
                cmd.append('--num_processes_segmentation_export')
                cmd.append(str(num_processes_segmentation_export))

            print(" ".join(cmd))
            subprocess.check_call(cmd)

def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PANORAMA PDAC inference workflow.")
    parser.add_argument("--input-dir", default=None, help="Directory containing input CT images.")
    parser.add_argument("--output-dir", default=None, help="Directory for challenge-style outputs.")
    parser.add_argument("--nnunet-base", default=None, help="Working directory for staged nnUNet inputs/outputs.")
    parser.add_argument("--model-dir", default=None, help="Root nnUNet_results directory containing model metadata and checkpoints.")
    parser.add_argument("--image-ext", default=".mha", help="Input image suffix to scan for, for example .mha or .nii.gz.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PDACDetectionContainer(
        nnunet_base=args.nnunet_base,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        image_ext=args.image_ext,
    ).process()
