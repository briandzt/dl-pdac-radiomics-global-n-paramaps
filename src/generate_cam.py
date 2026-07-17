#!/usr/bin/env python
"""
Class Activation Map (CAM) Generator for nnUNet with Radiomics Channels

Supports:
- GradCAM: Standard gradient-weighted CAM
- GradCAM++: Improved GradCAM with weighted gradients
- Input Gradient CAM: Gradient of output w.r.t. input channels

Generates CAM for tumor class (label 1) showing contribution of each input channel
(original CT + 8 radiomics feature maps).

Usage:
    python generate_cam.py --input_image /path/to/image.nii.gz \
                           --pancreas_mask /path/to/mask.nii.gz \
                           --output_dir /path/to/output \
                           --cam_type gradcam++
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage import binary_dilation

# Add src directory to path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# Import from existing codebase
from data_utils import (
    whole_panc_lbl, resample_img, CropPancreasROI, 
    GenerateFeatureROI, create_feature_specific_extractors
)
from voxel_radiomics import window_image_torch, load_config, extract_voxel_radiomics
from radiomics import featureextractor

# Import nnUNet components
sys.path.insert(0, str(SRC_DIR / "nnunetv2_global_rad"))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join


# Channel names for reference
CHANNEL_NAMES = {
    0: "CT",
    1: "feat_log-sigma-4-0-mm-3D_glcm_Correlation",
    2: "feat_wavelet-LLH_glszm_SizeZoneNonUniformityNormalized",
    3: "feat_wavelet-HLL_ngtdm_Strength",
    4: "feat_wavelet-HHL_glcm_Correlation",
    5: "feat_wavelet-LLL_glcm_Imc1",
    6: "feat_wavelet-LLL_glcm_Imc2",
    7: "feat_lbp-2D_glrlm_LongRunEmphasis",
    8: "feat_lbp-2D_glrlm_ShortRunEmphasis"
}


class CAMGenerator:
    """
    Class Activation Map generator for 3D medical image segmentation networks.
    Supports GradCAM, GradCAM++, and Input Gradient CAM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        self.input_gradients = None
        
        # Register hooks if target layer specified
        if target_layer is not None:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """
        Get the target layer for CAM computation.
        For UNet, we use the last encoder stage (bottleneck).
        """
        # For PlainConvUNet, the bottleneck is the last stage of encoder
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            if hasattr(encoder, 'stages'):
                return encoder.stages[-1]
        return None
    
    def gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        global_info: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute GradCAM for the specified target class.
        
        Args:
            input_tensor: Input tensor [B, C, D, H, W]
            target_class: Target class index (1 for tumor)
            global_info: Global radiomics features tensor
            
        Returns:
            CAM as numpy array [D, H, W]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if global_info is not None:
            global_info = global_info.to(self.device)
        
        # Forward pass
        if global_info is not None:
            output = self.model(input_tensor, global_info)
        else:
            output = self.model(input_tensor)
        
        # Handle deep supervision output
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        # Get probability for target class
        # Output shape: [B, num_classes, D, H, W]
        target_output = output[:, target_class, :, :, :]
        
        # Use sum of target class probabilities as the scalar target
        target = target_output.sum()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hook registration.")
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def gradcam_pp(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        global_info: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute GradCAM++ for the specified target class.
        
        GradCAM++ uses a weighted combination of positive gradients
        to provide better localization.
        
        Args:
            input_tensor: Input tensor [B, C, D, H, W]
            target_class: Target class index (1 for tumor)
            global_info: Global radiomics features tensor
            
        Returns:
            CAM as numpy array [D, H, W]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if global_info is not None:
            global_info = global_info.to(self.device)
        
        # Forward pass
        if global_info is not None:
            output = self.model(input_tensor, global_info)
        else:
            output = self.model(input_tensor)
        
        # Handle deep supervision output
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        # Get probability for target class
        target_output = output[:, target_class, :, :, :]
        target = target_output.sum()
        
        # First backward for first-order gradients
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")
        
        gradients = self.gradients.clone()
        activations = self.activations.clone()
        
        # GradCAM++ computation
        # Compute alpha (gradient weights)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Sum over spatial dimensions
        sum_activations = torch.sum(activations, dim=(2, 3, 4), keepdim=True)
        
        # Avoid division by zero
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = grad_2 / alpha_denom
        
        # Only keep positive gradients
        weights = alpha * F.relu(gradients)
        weights = torch.sum(weights, dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def input_gradient_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        global_info: Optional[torch.Tensor] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute Input Gradient CAM for each input channel.
        
        This shows how each input channel contributes to the target class prediction.
        
        Args:
            input_tensor: Input tensor [B, C, D, H, W]
            target_class: Target class index (1 for tumor)
            global_info: Global radiomics features tensor
            
        Returns:
            Dictionary mapping channel index to CAM array [D, H, W]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if global_info is not None:
            global_info = global_info.to(self.device)
        
        # Forward pass
        if global_info is not None:
            output = self.model(input_tensor, global_info)
        else:
            output = self.model(input_tensor)
        
        # Handle deep supervision output
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        # Get probability for target class
        target_output = output[:, target_class, :, :, :]
        target = target_output.sum()
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Get input gradients
        input_grads = input_tensor.grad.detach().cpu().numpy()
        
        # Compute CAM for each channel
        channel_cams = {}
        num_channels = input_tensor.shape[1]
        
        for c in range(num_channels):
            # Get gradient for this channel
            grad_c = input_grads[0, c, :, :, :]
            
            # Input gradient CAM: gradient * input (Gradient x Input)
            input_c = input_tensor[0, c, :, :, :].detach().cpu().numpy()
            cam = grad_c * input_c
            
            # Take absolute value for importance
            cam = np.abs(cam)
            
            # Normalize
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            
            channel_cams[c] = cam
        
        return channel_cams
    
    def generate_all_cams(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        global_info: Optional[torch.Tensor] = None,
        cam_type: str = 'gradcam++'
    ) -> Dict[str, Union[np.ndarray, Dict[int, np.ndarray]]]:
        """
        Generate all types of CAM for the input.
        
        Args:
            input_tensor: Input tensor [B, C, D, H, W]
            target_class: Target class index
            global_info: Global radiomics features
            cam_type: Primary CAM type ('gradcam', 'gradcam++', 'input_gradient')
            
        Returns:
            Dictionary with:
                - 'spatial_cam': Overall spatial CAM [D, H, W]
                - 'channel_cams': Per-channel CAMs {channel_idx: [D, H, W]}
                - 'combined_cam': Weighted combination of channel CAMs [D, H, W]
        """
        results = {}
        
        # Generate spatial CAM (GradCAM or GradCAM++)
        if cam_type == 'gradcam':
            spatial_cam = self.gradcam(input_tensor, target_class, global_info)
        elif cam_type == 'gradcam++':
            spatial_cam = self.gradcam_pp(input_tensor, target_class, global_info)
        else:
            spatial_cam = None
        
        results['spatial_cam'] = spatial_cam
        
        # Generate per-channel CAMs using input gradient method
        channel_cams = self.input_gradient_cam(input_tensor, target_class, global_info)
        results['channel_cams'] = channel_cams
        
        # Compute combined CAM (weighted average of channel CAMs)
        combined_cam = np.zeros_like(list(channel_cams.values())[0])
        for c, cam in channel_cams.items():
            combined_cam += cam
        combined_cam /= len(channel_cams)
        combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
        results['combined_cam'] = combined_cam
        
        return results


class CAMPipeline:
    """
    Complete pipeline for generating CAMs from raw NIfTI input.
    Uses the same preprocessing as process.py/process_local.py
    """
    
    def __init__(
        self,
        model_dir: str,
        voxel_config_path: str,
        device: str = 'cuda',
        folds: List[int] = [0],
        checkpoint_name: str = 'checkpoint_best.pth'
    ):
        self.model_dir = Path(model_dir)
        self.voxel_config_path = Path(voxel_config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        
        # Load model
        self.model, self.plans_manager, self.configuration_manager = self._load_model()
        self.model.eval()
        
        # Initialize CAM generator
        target_layer = self._get_bottleneck_layer()
        self.cam_generator = CAMGenerator(
            model=self.model,
            target_layer=target_layer,
            device=self.device
        )
        
        print(f"Model loaded from {model_dir}")
        print(f"Using device: {self.device}")
        print(f"Target layer for CAM: {type(target_layer).__name__}")
    
    def _load_model(self) -> Tuple[nn.Module, PlansManager, any]:
        """Load the trained nnUNet model."""
        # Load plans and dataset info
        dataset_json = load_json(join(str(self.model_dir), 'dataset.json'))
        plans = load_json(join(str(self.model_dir), 'plans.json'))
        plans_manager = PlansManager(plans)
        
        # Get configuration
        checkpoint = torch.load(
            join(str(self.model_dir), f'fold_{self.folds[0]}', self.checkpoint_name),
            map_location='cpu',
            weights_only=False
        )
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
        # Build network
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            'nnunetv2.training.nnUNetTrainer'
        )
        
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        
        # Load weights
        params = checkpoint['network_weights']
        
        # Enable attention if needed
        has_attn_keys = any(k.startswith('attn_layer.') for k in params.keys())
        if has_attn_keys and hasattr(network, 'enable_attention_from_dim'):
            if 'attn_layer.proj.weight' in params:
                global_dim = params['attn_layer.proj.weight'].shape[1]
                network.enable_attention_from_dim(global_dim)
        
        network.load_state_dict(params)
        network = network.to(self.device)
        
        return network, plans_manager, configuration_manager
    
    def _get_bottleneck_layer(self) -> nn.Module:
        """Get the bottleneck layer for GradCAM."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'stages'):
            return self.model.encoder.stages[-1]
        return None
    
    def preprocess_image(
        self,
        image_path: str,
        mask_path: str,
        output_dir: str
    ) -> Tuple[torch.Tensor, torch.Tensor, sitk.Image, Dict]:
        """
        Preprocess image following the same pipeline as process.py.
        
        Args:
            image_path: Path to input CT image
            mask_path: Path to pancreas mask
            output_dir: Directory for intermediate outputs
            
        Returns:
            Tuple of (input_tensor, global_info_tensor, reference_image, crop_coordinates)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read images
        itk_img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        mask_img = sitk.ReadImage(str(mask_path))
        
        # Process mask (combine pancreas labels)
        mask_img = whole_panc_lbl(mask_img)
        
        # Dilate mask
        mask_arr = sitk.GetArrayFromImage(mask_img)
        dilated_mask = binary_dilation(mask_arr, structure=np.ones((3, 3, 3), dtype=bool))
        mask_dilated = sitk.GetImageFromArray(dilated_mask.astype(int))
        mask_dilated.CopyInformation(mask_img)
        
        # Crop to pancreas ROI
        crop_margins = [100, 50, 15]
        cropped_image, crop_coordinates, _ = CropPancreasROI(itk_img, mask_dilated, crop_margins)
        cropped_label, _, _ = CropPancreasROI(mask_dilated, mask_dilated, crop_margins)
        cropped_label = resample_img(
            cropped_label,
            cropped_image.GetSpacing(),
            is_label=True,
            out_origin=cropped_image.GetOrigin(),
            out_size=cropped_image.GetSize()
        )
        
        # Resample for radiomics extraction
        samp_im = resample_img(cropped_image, [1.0, 1.0, 2.5], is_label=False, out_origin=cropped_image.GetOrigin())
        samp_msk = resample_img(
            cropped_label, samp_im.GetSpacing(),
            is_label=True, out_origin=samp_im.GetOrigin(), out_size=samp_im.GetSize()
        )
        
        # Extract voxel radiomics features
        basename = Path(image_path).stem.replace('.nii', '')
        temp_feature_dir = output_dir / f"temp_features_{basename}"
        temp_feature_dir.mkdir(exist_ok=True)
        
        feature_maps = self._extract_voxel_radiomics(
            samp_im, samp_msk, temp_feature_dir, basename
        )
        
        # Extract global radiomics features
        global_features = self._extract_global_radiomics(samp_im, samp_msk)
        
        # Resample feature maps back to cropped image space
        samp_imnew = resample_img(
            samp_im, cropped_image.GetSpacing(),
            is_label=False, out_origin=cropped_image.GetOrigin(), out_size=cropped_image.GetSize()
        )
        
        resampled_features = []
        for feature_path in feature_maps:
            feat_map = sitk.ReadImage(str(feature_path))
            feat_resampled = resample_img(
                feat_map, samp_imnew.GetSpacing(),
                is_label=False, out_origin=samp_imnew.GetOrigin(), out_size=samp_imnew.GetSize()
            )
            resampled_features.append(sitk.GetArrayFromImage(feat_resampled))
        
        # Clean up temp directory
        shutil.rmtree(temp_feature_dir, ignore_errors=True)
        
        # Build input tensor (original image + feature maps)
        original_arr = sitk.GetArrayFromImage(cropped_image)
        input_channels = [original_arr] + resampled_features
        input_array = np.stack(input_channels, axis=0)  # [C, D, H, W]
        input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)  # [1, C, D, H, W]
        
        # Build global info tensor
        global_tensor = torch.from_numpy(global_features).float()
        
        return input_tensor, global_tensor, cropped_image, crop_coordinates
    
    def _extract_voxel_radiomics(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        output_dir: Path,
        basename: str
    ) -> List[Path]:
        """Extract voxel-based radiomics features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_dir = temp_path / "images"
            mask_dir = temp_path / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            
            # Save temporary files
            sitk.WriteImage(image, str(image_dir / f"{basename}.nii.gz"))
            sitk.WriteImage(mask, str(mask_dir / f"{basename}.nii.gz"))
            
            # Load and modify config
            config = load_config(str(self.voxel_config_path))
            config['IO'] = {
                'image_dir': str(image_dir),
                'mask_dir': str(mask_dir),
                'output': str(output_dir),
                'nnunet_format': True
            }
            
            temp_config = temp_path / "temp_config.json"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=4)
            
            extract_voxel_radiomics(str(temp_config))
            
            # Collect feature map paths
            feature_maps = []
            for i in range(1, len(config['features']) + 1):
                feat_path = output_dir / f"{basename}_{i:04d}.nii.gz"
                if feat_path.exists():
                    feature_maps.append(feat_path)
            
            return feature_maps
    
    def _extract_global_radiomics(
        self,
        image: sitk.Image,
        mask: sitk.Image
    ) -> np.ndarray:
        """Extract global radiomics features."""
        features = [
            'feat_log-sigma-3-0-mm-3D_glcm_Correlation',
            'feat_wavelet-LLH_glcm_Imc1',
            'feat_wavelet-HLL_glszm_SmallAreaEmphasis',
            'feat_wavelet-HHL_firstorder_Mean',
            'feat_wavelet-HHL_glcm_Correlation',
            'feat_wavelet-LLL_glcm_MCC',
            'feat_gradient_glcm_Imc1',
            'feat_lbp-2D_glrlm_ShortRunEmphasis',
            'feat_original_shape_Sphericity',
            'feat_original_shape_SurfaceVolumeRatio'
        ]
        
        config = load_config(str(self.voxel_config_path))
        settings = config['settings']
        
        # Resample and window
        img = resample_img(image, settings['target_spacing'], is_label=False)
        msk = resample_img(mask, settings['target_spacing'], is_label=True)
        
        window_min = settings['window_level'] - settings['window_width'] / 2
        window_max = settings['window_level'] + settings['window_width'] / 2
        img = window_image_torch(img, window_min, window_max, device='cpu', to_int=True)
        
        # Create extractor
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
            result = extractor.execute(img, msk, voxelBased=False)
            for key in result.keys():
                if 'diagnostics' not in key:
                    global_features.append(float(result[key]))
                    break
        
        return np.array(global_features).reshape(1, -1)
    
    def generate_cams(
        self,
        image_path: str,
        mask_path: str,
        output_dir: str,
        cam_types: List[str] = ['gradcam++', 'input_gradient'],
        target_class: int = 1
    ) -> Dict[str, Dict]:
        """
        Generate CAMs for the input image.
        
        Args:
            image_path: Path to input CT image
            mask_path: Path to pancreas mask
            output_dir: Output directory for CAM files
            cam_types: List of CAM types to generate
            target_class: Target class index (1 for tumor)
            
        Returns:
            Dictionary with CAM results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {image_path}")
        print(f"Mask: {mask_path}")
        
        # Preprocess
        print("Preprocessing...")
        input_tensor, global_tensor, ref_image, crop_coords = self.preprocess_image(
            image_path, mask_path, str(output_dir / "temp")
        )
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Global features shape: {global_tensor.shape}")
        
        results = {}
        basename = Path(image_path).stem.replace('.nii', '')
        
        # Generate CAMs for each type
        for cam_type in cam_types:
            print(f"\nGenerating {cam_type.upper()} CAMs...")
            
            if cam_type in ['gradcam', 'gradcam++']:
                # Spatial CAM from GradCAM/GradCAM++
                cam_results = self.cam_generator.generate_all_cams(
                    input_tensor, target_class, global_tensor, cam_type
                )
                
                # Save spatial CAM
                if cam_results['spatial_cam'] is not None:
                    spatial_cam_img = sitk.GetImageFromArray(cam_results['spatial_cam'].astype(np.float32))
                    spatial_cam_img.CopyInformation(ref_image)
                    cam_path = output_dir / f"{basename}_{cam_type}_spatial.nii.gz"
                    sitk.WriteImage(spatial_cam_img, str(cam_path))
                    print(f"  Saved: {cam_path}")
                
                results[f'{cam_type}_spatial'] = cam_results['spatial_cam']
                
            elif cam_type == 'input_gradient':
                # Per-channel CAMs
                channel_cams = self.cam_generator.input_gradient_cam(
                    input_tensor, target_class, global_tensor
                )
                
                # Save each channel CAM
                for c, cam in channel_cams.items():
                    channel_name = CHANNEL_NAMES.get(c, f"channel_{c}")
                    safe_name = channel_name.replace('-', '_').replace(' ', '_')
                    
                    cam_img = sitk.GetImageFromArray(cam.astype(np.float32))
                    cam_img.CopyInformation(ref_image)
                    cam_path = output_dir / f"{basename}_input_grad_ch{c}_{safe_name}.nii.gz"
                    sitk.WriteImage(cam_img, str(cam_path))
                    print(f"  Saved: {cam_path}")
                
                # Save combined channel CAM
                combined_cam = np.zeros_like(list(channel_cams.values())[0])
                for cam in channel_cams.values():
                    combined_cam += cam
                combined_cam /= len(channel_cams)
                combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
                
                combined_img = sitk.GetImageFromArray(combined_cam.astype(np.float32))
                combined_img.CopyInformation(ref_image)
                combined_path = output_dir / f"{basename}_input_grad_combined.nii.gz"
                sitk.WriteImage(combined_img, str(combined_path))
                print(f"  Saved: {combined_path}")
                
                results['input_gradient_channels'] = channel_cams
                results['input_gradient_combined'] = combined_cam
        
        # Save channel importance summary
        if 'input_gradient_channels' in results:
            importance = {}
            for c, cam in results['input_gradient_channels'].items():
                importance[CHANNEL_NAMES.get(c, f"channel_{c}")] = float(cam.mean())
            
            summary_path = output_dir / f"{basename}_channel_importance.json"
            with open(summary_path, 'w') as f:
                json.dump(importance, f, indent=4)
            print(f"\nChannel importance saved: {summary_path}")
        
        # Clean up temp directory
        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"\nCAM generation complete. Results saved to: {output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate Class Activation Maps for nnUNet with radiomics channels'
    )
    parser.add_argument(
        '--input_image', '-i', type=str, required=True,
        help='Path to input CT image (NIfTI format)'
    )
    parser.add_argument(
        '--pancreas_mask', '-m', type=str, required=True,
        help='Path to pancreas segmentation mask (NIfTI format)'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, required=True,
        help='Output directory for CAM files'
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=str(SRC_DIR / "nnUNet_results" / "Dataset002_stage2"),
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--voxel_config', type=str,
        default=str(SRC_DIR / "PANORAMA_voxel.json"),
        help='Path to voxel radiomics configuration'
    )
    parser.add_argument(
        '--cam_types', nargs='+', type=str,
        default=['gradcam++', 'input_gradient'],
        choices=['gradcam', 'gradcam++', 'input_gradient'],
        help='CAM types to generate'
    )
    parser.add_argument(
        '--target_class', type=int, default=1,
        help='Target class index (1 for tumor)'
    )
    parser.add_argument(
        '--folds', nargs='+', type=int, default=[0],
        help='Model folds to use'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoint_best.pth',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CAMPipeline(
        model_dir=args.model_dir,
        voxel_config_path=args.voxel_config,
        device=args.device,
        folds=args.folds,
        checkpoint_name=args.checkpoint
    )
    
    # Generate CAMs
    results = pipeline.generate_cams(
        image_path=args.input_image,
        mask_path=args.pancreas_mask,
        output_dir=args.output_dir,
        cam_types=args.cam_types,
        target_class=args.target_class
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

