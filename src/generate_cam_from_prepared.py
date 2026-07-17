#!/usr/bin/env python
"""
Generate Class Activation Maps from pre-prepared nnUNet inputs.

Use this script when you already have the 9-channel input prepared:
- casename_0000.nii.gz (CT image)
- casename_0001.nii.gz through casename_0008.nii.gz (radiomics feature maps)
- casename_global.npy (global radiomics features)

Usage:
    python generate_cam_from_prepared.py \
        --input_dir /path/to/prepared/inputs \
        --casename PANORAMA_0001 \
        --output_dir /path/to/output \
        --cam_types gradcam++ input_gradient
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src directory to path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "nnunetv2_global_rad"))

# Import from generate_cam
from generate_cam import CAMGenerator, CHANNEL_NAMES

# Import nnUNet components
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join


class PreparedInputCAMPipeline:
    """
    CAM generation pipeline for pre-prepared nnUNet inputs.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = 'cuda',
        folds: List[int] = [0],
        checkpoint_name: str = 'checkpoint_best.pth'
    ):
        self.model_dir = Path(model_dir)
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
    
    def _load_model(self) -> Tuple[nn.Module, PlansManager, any]:
        """Load the trained nnUNet model."""
        dataset_json = load_json(join(str(self.model_dir), 'dataset.json'))
        plans = load_json(join(str(self.model_dir), 'plans.json'))
        plans_manager = PlansManager(plans)
        
        checkpoint = torch.load(
            join(str(self.model_dir), f'fold_{self.folds[0]}', self.checkpoint_name),
            map_location='cpu',
            weights_only=False
        )
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
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
        
        params = checkpoint['network_weights']
        
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
    
    def load_prepared_input(
        self,
        input_dir: str,
        casename: str,
        num_channels: int = 9
    ) -> Tuple[torch.Tensor, torch.Tensor, sitk.Image]:
        """
        Load pre-prepared nnUNet input files.
        
        Args:
            input_dir: Directory containing prepared inputs
            casename: Case identifier (e.g., 'PANORAMA_0001')
            num_channels: Number of input channels (9 = CT + 8 radiomics)
            
        Returns:
            Tuple of (input_tensor, global_info_tensor, reference_image)
        """
        input_dir = Path(input_dir)
        
        # Load all channels
        channels = []
        ref_image = None
        
        for i in range(num_channels):
            channel_path = input_dir / f"{casename}_{i:04d}.nii.gz"
            if not channel_path.exists():
                raise FileNotFoundError(f"Channel file not found: {channel_path}")
            
            img = sitk.ReadImage(str(channel_path))
            arr = sitk.GetArrayFromImage(img)
            channels.append(arr)
            
            if i == 0:
                ref_image = img
        
        # Stack channels
        input_array = np.stack(channels, axis=0)  # [C, D, H, W]
        input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)  # [1, C, D, H, W]
        
        # Load global features
        global_path = input_dir / f"{casename}_global.npy"
        if global_path.exists():
            global_features = np.load(str(global_path))
            global_tensor = torch.from_numpy(global_features).float()
        else:
            print(f"Warning: Global features not found at {global_path}, using zeros")
            global_tensor = torch.zeros(1, 10)  # Default 10 global features
        
        print(f"Loaded input: {input_tensor.shape}")
        print(f"Global features: {global_tensor.shape}")
        
        return input_tensor, global_tensor, ref_image
    
    def generate_cams(
        self,
        input_dir: str,
        casename: str,
        output_dir: str,
        cam_types: List[str] = ['gradcam++', 'input_gradient'],
        target_class: int = 1
    ) -> Dict[str, Dict]:
        """
        Generate CAMs for the prepared input.
        
        Args:
            input_dir: Directory containing prepared inputs
            casename: Case identifier
            output_dir: Output directory for CAM files
            cam_types: List of CAM types to generate
            target_class: Target class index (1 for tumor)
            
        Returns:
            Dictionary with CAM results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing case: {casename}")
        
        # Load prepared input
        input_tensor, global_tensor, ref_image = self.load_prepared_input(
            input_dir, casename
        )
        
        results = {}
        
        # Generate CAMs for each type
        for cam_type in cam_types:
            print(f"\nGenerating {cam_type.upper()} CAMs...")
            
            if cam_type in ['gradcam', 'gradcam++']:
                cam_results = self.cam_generator.generate_all_cams(
                    input_tensor, target_class, global_tensor, cam_type
                )
                
                if cam_results['spatial_cam'] is not None:
                    spatial_cam_img = sitk.GetImageFromArray(
                        cam_results['spatial_cam'].astype(np.float32)
                    )
                    spatial_cam_img.CopyInformation(ref_image)
                    cam_path = output_dir / f"{casename}_{cam_type}_spatial.nii.gz"
                    sitk.WriteImage(spatial_cam_img, str(cam_path))
                    print(f"  Saved: {cam_path}")
                
                results[f'{cam_type}_spatial'] = cam_results['spatial_cam']
                
            elif cam_type == 'input_gradient':
                channel_cams = self.cam_generator.input_gradient_cam(
                    input_tensor, target_class, global_tensor
                )
                
                # Save each channel CAM
                for c, cam in channel_cams.items():
                    channel_name = CHANNEL_NAMES.get(c, f"channel_{c}")
                    safe_name = channel_name.replace('-', '_').replace(' ', '_')
                    
                    cam_img = sitk.GetImageFromArray(cam.astype(np.float32))
                    cam_img.CopyInformation(ref_image)
                    cam_path = output_dir / f"{casename}_input_grad_ch{c}_{safe_name}.nii.gz"
                    sitk.WriteImage(cam_img, str(cam_path))
                    print(f"  Saved: {cam_path}")
                
                # Combined CAM
                combined_cam = np.zeros_like(list(channel_cams.values())[0])
                for cam in channel_cams.values():
                    combined_cam += cam
                combined_cam /= len(channel_cams)
                combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
                
                combined_img = sitk.GetImageFromArray(combined_cam.astype(np.float32))
                combined_img.CopyInformation(ref_image)
                combined_path = output_dir / f"{casename}_input_grad_combined.nii.gz"
                sitk.WriteImage(combined_img, str(combined_path))
                print(f"  Saved: {combined_path}")
                
                results['input_gradient_channels'] = channel_cams
                results['input_gradient_combined'] = combined_cam
        
        # Save channel importance summary
        if 'input_gradient_channels' in results:
            importance = {}
            for c, cam in results['input_gradient_channels'].items():
                importance[CHANNEL_NAMES.get(c, f"channel_{c}")] = float(cam.mean())
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            summary_path = output_dir / f"{casename}_channel_importance.json"
            with open(summary_path, 'w') as f:
                json.dump(importance, f, indent=4)
            print(f"\nChannel importance saved: {summary_path}")
        
        print(f"\nCAM generation complete. Results saved to: {output_dir}")
        
        return results
    
    def generate_cams_batch(
        self,
        input_dir: str,
        output_dir: str,
        cam_types: List[str] = ['gradcam++', 'input_gradient'],
        target_class: int = 1
    ) -> Dict[str, Dict]:
        """
        Generate CAMs for all cases in the input directory.
        
        Args:
            input_dir: Directory containing prepared inputs
            output_dir: Output directory for CAM files
            cam_types: List of CAM types to generate
            target_class: Target class index
            
        Returns:
            Dictionary mapping casename to CAM results
        """
        input_dir = Path(input_dir)
        
        # Find all cases by looking for _0000.nii.gz files
        case_files = list(input_dir.glob("*_0000.nii.gz"))
        casenames = [f.stem.replace("_0000", "") for f in case_files]
        
        print(f"Found {len(casenames)} cases to process")
        
        all_results = {}
        for casename in casenames:
            try:
                results = self.generate_cams(
                    input_dir=str(input_dir),
                    casename=casename,
                    output_dir=output_dir,
                    cam_types=cam_types,
                    target_class=target_class
                )
                all_results[casename] = results
            except Exception as e:
                print(f"Error processing {casename}: {e}")
                continue
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Generate CAMs from pre-prepared nnUNet inputs'
    )
    parser.add_argument(
        '--input_dir', '-i', type=str, required=True,
        help='Directory containing prepared inputs (casename_XXXX.nii.gz files)'
    )
    parser.add_argument(
        '--casename', '-c', type=str, default=None,
        help='Specific case to process. If not provided, processes all cases in input_dir'
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
    pipeline = PreparedInputCAMPipeline(
        model_dir=args.model_dir,
        device=args.device,
        folds=args.folds,
        checkpoint_name=args.checkpoint
    )
    
    if args.casename:
        # Process single case
        results = pipeline.generate_cams(
            input_dir=args.input_dir,
            casename=args.casename,
            output_dir=args.output_dir,
            cam_types=args.cam_types,
            target_class=args.target_class
        )
    else:
        # Process all cases in directory
        results = pipeline.generate_cams_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cam_types=args.cam_types,
            target_class=args.target_class
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

