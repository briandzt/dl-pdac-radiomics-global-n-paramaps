#!/usr/bin/env python
"""
Unified PDAC analysis pipeline for nnUNet with voxel/global radiomics.

Combines radiomics feature extraction, CAM generation, and segmentation
into a single configurable workflow.

Outputs (controlled via flags):
  --radiomics   Save resampled voxel radiomics feature maps + global features
  --cam         Generate GradCAM++ and Input Gradient CAMs
  --segmentation Run nnUNetv2_predict for multi-label segmentation + detection map

At least one of --radiomics, --cam, --segmentation is required.

Pancreas mask:
  If --pancreas_mask is provided, use it directly.
  Otherwise, run the Stage-1 lowres model (Dataset001_LR) to generate one.

Usage examples:

    # Everything (auto lowres mask):
    python pipeline.py -i ct_0000.nii.gz -o ./results --radiomics --cam --segmentation

    # CAM + segmentation with existing mask:
    python pipeline.py -i ct_0000.nii.gz -m mask.nii.gz -o ./results --cam --segmentation

    # Only radiomics feature maps:
    python pipeline.py -i ct_0000.nii.gz -m mask.nii.gz -o ./results --radiomics
"""

import os
import sys
import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "nnunetv2_global_rad"))

from data_utils import (
    whole_panc_lbl, resample_img, CropPancreasROI,
    GetFullSizDetectionMap, GetFullSizDetectionMap_nifti,
    PostProcessing, create_feature_specific_extractors,
)
from voxel_radiomics import window_image_torch, load_config, extract_voxel_radiomics
from radiomics import featureextractor

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# ───────────────────────────── constants ──────────────────────────────

CHANNEL_NAMES = {
    0: "CT",
    1: "feat_log-sigma-4-0-mm-3D_glcm_Correlation",
    2: "feat_wavelet-LLH_glszm_SizeZoneNonUniformityNormalized",
    3: "feat_wavelet-HLL_ngtdm_Strength",
    4: "feat_wavelet-HHL_glcm_Correlation",
    5: "feat_wavelet-LLL_glcm_Imc1",
    6: "feat_wavelet-LLL_glcm_Imc2",
    7: "feat_lbp-2D_glrlm_LongRunEmphasis",
    8: "feat_lbp-2D_glrlm_ShortRunEmphasis",
}

GLOBAL_FEATURE_NAMES = [
    'feat_log-sigma-3-0-mm-3D_glcm_Correlation',
    'feat_wavelet-LLH_glcm_Imc1',
    'feat_wavelet-HLL_glszm_SmallAreaEmphasis',
    'feat_wavelet-HHL_firstorder_Mean',
    'feat_wavelet-HHL_glcm_Correlation',
    'feat_wavelet-LLL_glcm_MCC',
    'feat_gradient_glcm_Imc1',
    'feat_lbp-2D_glrlm_ShortRunEmphasis',
    'feat_original_shape_Sphericity',
    'feat_original_shape_SurfaceVolumeRatio',
]

# ───────────────────────── CAM generator ──────────────────────────────

class CAMGenerator:
    """GradCAM / GradCAM++ / Input-Gradient CAM for 3-D segmentation nets."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module],
                 device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

        if target_layer is not None:
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, 'activations', o.detach()))
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def _forward(self, x, global_info):
        x = x.to(self.device).requires_grad_(True)
        g = global_info.to(self.device) if global_info is not None else None
        out = self.model(x, g) if g is not None else self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return x, out

    def gradcam(self, input_tensor, target_class=1, global_info=None):
        self.model.eval()
        inp, out = self._forward(input_tensor, global_info)
        target = out[:, target_class].sum()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=inp.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def gradcam_pp(self, input_tensor, target_class=1, global_info=None):
        self.model.eval()
        inp, out = self._forward(input_tensor, global_info)
        target = out[:, target_class].sum()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        g, a = self.gradients.clone(), self.activations.clone()
        g2, g3 = g ** 2, g ** 3
        alpha = g2 / (2 * g2 + a.sum(dim=(2, 3, 4), keepdim=True) * g3 + 1e-8)
        weights = (alpha * F.relu(g)).sum(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=inp.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def input_gradient_cam(self, input_tensor, target_class=1, global_info=None):
        self.model.eval()
        inp, out = self._forward(input_tensor, global_info)
        self.model.zero_grad()
        out[:, target_class].sum().backward()
        grads = inp.grad.detach().cpu().numpy()
        inputs = inp.detach().cpu().numpy()
        cams = {}
        for c in range(inp.shape[1]):
            cam = np.abs(grads[0, c] * inputs[0, c])
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            cams[c] = cam
        return cams

# ────────────────────── radiomics helpers ─────────────────────────────

def _extract_voxel_radiomics(image, mask, output_dir, config_path, basename):
    """Extract voxel-based radiomics features → list of NIfTI paths."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_dir, msk_dir = tmp / "images", tmp / "masks"
        img_dir.mkdir(); msk_dir.mkdir()
        sitk.WriteImage(image, str(img_dir / f"{basename}.nii.gz"))
        sitk.WriteImage(mask, str(msk_dir / f"{basename}.nii.gz"))

        with open(config_path, 'r') as f:
            config = json.load(f)
        config['IO'] = {'image_dir': str(img_dir), 'mask_dir': str(msk_dir),
                         'output': str(output_dir), 'nnunet_format': True}
        cfg_path = tmp / "cfg.json"
        with open(cfg_path, 'w') as f:
            json.dump(config, f, indent=4)

        extract_voxel_radiomics(str(cfg_path))

        return [Path(output_dir) / f"{basename}_{i:04d}.nii.gz"
                for i in range(1, len(config['features']) + 1)
                if (Path(output_dir) / f"{basename}_{i:04d}.nii.gz").exists()]


def _extract_global_radiomics(image, mask, config_path):
    """Extract 10 global radiomics scalars → (1, 10) ndarray."""
    config = load_config(config_path)
    s = config['settings']
    img = resample_img(image, s['target_spacing'], is_label=False)
    msk = resample_img(mask, s['target_spacing'], is_label=True)
    w_lo = s['window_level'] - s['window_width'] / 2
    w_hi = s['window_level'] + s['window_width'] / 2
    img = window_image_torch(img, w_lo, w_hi, device='cpu', to_int=True)

    base = featureextractor.RadiomicsFeatureExtractor(
        voxelBased=False, padDistance=10, dtype=torch.float32,
        device='cpu', binWidth=10)
    base.settings['label'] = 1
    extractors = create_feature_specific_extractors(base, np.array(GLOBAL_FEATURE_NAMES))

    vals = []
    for name, ext in extractors:
        print(f"  Global: {name}")
        res = ext.execute(img, msk, voxelBased=False)
        for k in res:
            if 'diagnostics' not in k:
                vals.append(float(res[k])); break
    return np.array(vals).reshape(1, -1)

# ────────────── nnUNet predict (CLI) helpers ──────────────────────────

def _run_nnunet_predict(input_dir, output_dir, model_dir, task,
                        trainer="nnUNetTrainer", checkpoint="checkpoint_best.pth",
                        folds="0,1,2,3,4", save_probs=True, global_info_dir=None):
    os.environ['RESULTS_FOLDER'] = str(model_dir)
    cmd = ['nnUNetv2_predict', '-d', task, '-i', str(input_dir), '-o', str(output_dir),
           '-c', '3d_fullres', '-tr', trainer, '--disable_progress_bar',
           '--continue_prediction', '-f', *folds.split(','), '-chk', checkpoint]
    if save_probs:
        cmd.append('--save_probabilities')
    if global_info_dir:
        cmd.extend(['-global_info_folder', str(global_info_dir)])
    print(f"CMD: {' '.join(cmd)}")
    subprocess.check_call(' '.join(cmd), shell=True)

# ─────────────── model loader (for CAM forward pass) ─────────────────

def _resolve_trainer_dir(model_dir):
    """Resolve the trainer subdirectory inside a dataset dir (nnUNet layout)."""
    model_dir = Path(model_dir)
    # If dataset.json is directly here and there's a trainer subdir, descend into it
    trainer_dirs = [d for d in model_dir.iterdir()
                    if d.is_dir() and '__nnUNetPlans__' in d.name]
    if trainer_dirs:
        return trainer_dirs[0]
    return model_dir


def _load_model(model_dir, folds=(0,), checkpoint_name="checkpoint_best.pth",
                device="cuda"):
    model_dir = _resolve_trainer_dir(Path(model_dir))
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset_json = load_json(join(str(model_dir), 'dataset.json'))
    plans = load_json(join(str(model_dir), 'plans.json'))
    pm = PlansManager(plans)

    ckpt = torch.load(join(str(model_dir), f'fold_{folds[0]}', checkpoint_name),
                       map_location='cpu', weights_only=False)
    cm = pm.get_configuration(ckpt['init_args']['configuration'])
    n_in = determine_num_input_channels(pm, cm, dataset_json)
    n_out = pm.get_label_manager(dataset_json).num_segmentation_heads
    trainer_cls = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        ckpt['trainer_name'], 'nnunetv2.training.nnUNetTrainer')
    net = trainer_cls.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, n_in, n_out,
        enable_deep_supervision=False)
    params = ckpt['network_weights']
    if any(k.startswith('attn_layer.') for k in params):
        if 'attn_layer.proj.weight' in params and hasattr(net, 'enable_attention_from_dim'):
            net.enable_attention_from_dim(params['attn_layer.proj.weight'].shape[1])
    net.load_state_dict(params)
    net.to(device).eval()
    return net, device

# ═══════════════════════ main pipeline ════════════════════════════════

def run_pipeline(
    input_image: str,
    output_dir: str,
    pancreas_mask: Optional[str] = None,
    do_radiomics: bool = False,
    do_cam: bool = False,
    do_segmentation: bool = False,
    model_dir: str = str(SRC_DIR / "nnUNet_results"),
    voxel_config: str = str(SRC_DIR / "PANORAMA_voxel.json"),
    cam_types: List[str] = None,
    cam_target_class: int = 1,
    cam_folds: List[int] = None,
    cam_device: str = "cuda",
    seg_trainer: str = "nnUNetTrainer_Loss_CE_checkpoints",
    seg_checkpoint: str = "checkpoint_best.pth",
    seg_folds: str = "0,1,2,3,4",
):
    if cam_types is None:
        cam_types = ['gradcam++', 'input_gradient']
    if cam_folds is None:
        cam_folds = [0]

    input_image = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    stage2_dir = model_dir / "Dataset002_stage2"

    basename = input_image.name.replace('.nii.gz', '').replace('.mha', '')
    casename = basename[:-5] if basename.endswith('_0000') else basename

    itk_img = sitk.ReadImage(str(input_image), sitk.sitkFloat32)
    print(f"[input]  {input_image}  size={itk_img.GetSize()}")

    # ── Stage 1: pancreas mask ────────────────────────────────────────
    if pancreas_mask is not None:
        print(f"[mask]   provided: {pancreas_mask}")
        mask_base = whole_panc_lbl(sitk.ReadImage(str(pancreas_mask)))
    else:
        print("[mask]   running lowres model ...")
        with tempfile.TemporaryDirectory() as tmp:
            lr_in, lr_out = Path(tmp) / "lr_in", Path(tmp) / "lr_out"
            lr_in.mkdir(); lr_out.mkdir()
            lr_img = resample_img(itk_img, (4.5, 4.5, 9.0), is_label=False,
                                  out_origin=itk_img.GetOrigin(), out_size=[])
            sitk.WriteImage(lr_img, str(lr_in / f"{basename}.nii.gz"))
            _run_nnunet_predict(lr_in, lr_out, model_dir, task="001",
                                trainer="nnUNetTrainer", folds="0,1",
                                save_probs=False)
            mask_base = whole_panc_lbl(
                sitk.ReadImage(str(lr_out / f"{casename}.nii.gz")))

    mask_arr = sitk.GetArrayFromImage(mask_base)
    if mask_arr.sum() == 0:
        raise ValueError("Pancreas mask is empty.")

    dilated = binary_dilation(mask_arr, structure=np.ones((3, 3, 3), dtype=bool))
    mask_dilated = sitk.GetImageFromArray(dilated.astype(int))
    mask_dilated.CopyInformation(mask_base)

    # ── Crop pancreas ROI ─────────────────────────────────────────────
    print("[crop]   extracting pancreas ROI ...")
    crop_margins = [100, 50, 15]
    cropped_image, crop_coords, _ = CropPancreasROI(itk_img, mask_dilated, crop_margins)
    cropped_label, _, _ = CropPancreasROI(mask_dilated, mask_dilated, crop_margins)
    cropped_label = resample_img(cropped_label, cropped_image.GetSpacing(),
                                 is_label=True,
                                 out_origin=cropped_image.GetOrigin(),
                                 out_size=cropped_image.GetSize())

    # ── Radiomics extraction (always needed) ──────────────────────────
    print("[rad]    extracting voxel radiomics ...")
    samp_im = resample_img(cropped_image, [1.0, 1.0, 2.5], is_label=False,
                           out_origin=cropped_image.GetOrigin())
    samp_msk = resample_img(cropped_label, samp_im.GetSpacing(), is_label=True,
                            out_origin=samp_im.GetOrigin(), out_size=samp_im.GetSize())

    with tempfile.TemporaryDirectory() as feat_tmp:
        feat_tmp = Path(feat_tmp)
        voxel_maps = _extract_voxel_radiomics(samp_im, samp_msk, str(feat_tmp),
                                               voxel_config, casename)

        print("[rad]    extracting global radiomics ...")
        global_features = _extract_global_radiomics(samp_im, samp_msk, voxel_config)

        # Resample feature maps back to cropped-image space
        samp_back = resample_img(samp_im, cropped_image.GetSpacing(), is_label=False,
                                 out_origin=cropped_image.GetOrigin(),
                                 out_size=cropped_image.GetSize())
        resampled_arrays = []
        resampled_images = []
        for fp in voxel_maps:
            fm = sitk.ReadImage(str(fp))
            fm_rs = resample_img(fm, samp_back.GetSpacing(), is_label=False,
                                 out_origin=samp_back.GetOrigin(),
                                 out_size=samp_back.GetSize())
            resampled_arrays.append(sitk.GetArrayFromImage(fm_rs))
            resampled_images.append(fm_rs)

    # ── Save radiomics outputs ────────────────────────────────────────
    if do_radiomics:
        rad_dir = output_dir / "radiomics"
        rad_dir.mkdir(exist_ok=True)
        sitk.WriteImage(cropped_image, str(rad_dir / f"{casename}_0000.nii.gz"))
        for i, fm_img in enumerate(resampled_images):
            sitk.WriteImage(fm_img, str(rad_dir / f"{casename}_{i+1:04d}.nii.gz"))
        np.save(str(rad_dir / f"{casename}_global.npy"), global_features)
        print(f"[rad]    saved {len(resampled_images)+1} channels + global → {rad_dir}")

    # ── CAM generation ────────────────────────────────────────────────
    if do_cam:
        cam_dir = output_dir / "cam"
        cam_dir.mkdir(exist_ok=True)

        print("[cam]    loading model for CAM ...")
        net, dev = _load_model(stage2_dir, folds=cam_folds, device=cam_device)
        bottleneck = (net.encoder.stages[-1]
                      if hasattr(net, 'encoder') and hasattr(net.encoder, 'stages')
                      else None)
        cam_gen = CAMGenerator(net, bottleneck, dev)

        ct_arr = sitk.GetArrayFromImage(cropped_image)
        inp = np.stack([ct_arr] + resampled_arrays, axis=0)
        inp_t = torch.from_numpy(inp).float().unsqueeze(0)
        glob_t = torch.from_numpy(global_features).float()

        ref_image = cropped_image

        for cam_type in cam_types:
            print(f"[cam]    generating {cam_type} ...")

            if cam_type in ('gradcam', 'gradcam++'):
                fn = cam_gen.gradcam if cam_type == 'gradcam' else cam_gen.gradcam_pp
                spatial = fn(inp_t, cam_target_class, glob_t)
                img_out = sitk.GetImageFromArray(spatial.astype(np.float32))
                img_out.CopyInformation(ref_image)
                p = cam_dir / f"{casename}_{cam_type}_spatial.nii.gz"
                sitk.WriteImage(img_out, str(p))
                print(f"           → {p}")

            if cam_type == 'input_gradient':
                ch_cams = cam_gen.input_gradient_cam(inp_t, cam_target_class, glob_t)
                for c, cam in ch_cams.items():
                    name = CHANNEL_NAMES.get(c, f"channel_{c}").replace('-', '_').replace(' ', '_')
                    img_out = sitk.GetImageFromArray(cam.astype(np.float32))
                    img_out.CopyInformation(ref_image)
                    p = cam_dir / f"{casename}_input_grad_ch{c}_{name}.nii.gz"
                    sitk.WriteImage(img_out, str(p))
                    print(f"           → {p}")

                combined = np.mean(list(ch_cams.values()), axis=0)
                combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
                img_out = sitk.GetImageFromArray(combined.astype(np.float32))
                img_out.CopyInformation(ref_image)
                p = cam_dir / f"{casename}_input_grad_combined.nii.gz"
                sitk.WriteImage(img_out, str(p))
                print(f"           → {p}")

                importance = {CHANNEL_NAMES.get(c, f"ch{c}"): float(ch_cams[c].mean())
                              for c in sorted(ch_cams)}
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                p = cam_dir / f"{casename}_channel_importance.json"
                with open(p, 'w') as f:
                    json.dump(importance, f, indent=4)
                print(f"           → {p}")

        del net, cam_gen
        torch.cuda.empty_cache()
        print(f"[cam]    done → {cam_dir}")

    # ── Segmentation via nnUNetv2_predict ─────────────────────────────
    if do_segmentation:
        print("[seg]    preparing nnUNet input ...")
        with tempfile.TemporaryDirectory() as tmp:
            fr_in, fr_out = Path(tmp) / "fr_in", Path(tmp) / "fr_out"
            fr_in.mkdir(); fr_out.mkdir()

            sitk.WriteImage(cropped_image, str(fr_in / f"{casename}_0000.nii.gz"))
            for i, fm_img in enumerate(resampled_images):
                sitk.WriteImage(fm_img, str(fr_in / f"{casename}_{i+1:04d}.nii.gz"))
            np.save(str(fr_in / f"{casename}_global.npy"), global_features)

            print("[seg]    running fullres nnUNet ...")
            _run_nnunet_predict(fr_in, fr_out, model_dir, task="002",
                                trainer=seg_trainer, checkpoint=seg_checkpoint,
                                folds=seg_folds, save_probs=True,
                                global_info_dir=fr_in)

            seg_dir = output_dir / "segmentation"
            seg_dir.mkdir(exist_ok=True)

            npz_path = str(fr_out / f"{casename}.npz")
            nii_path = str(fr_out / f"{casename}.nii.gz")

            prediction = np.load(npz_path)
            pred_post = PostProcessing(prediction, nii_path)
            det_map, pdac_score = GetFullSizDetectionMap(pred_post, crop_coords, itk_img)

            p = seg_dir / f"{casename}_detection_map.mha"
            sitk.WriteImage(det_map, str(p))
            print(f"[seg]    → {p}")

            p = seg_dir / f"{casename}_pdac-likelihood.json"
            with open(p, 'w') as f:
                json.dump(pdac_score, f, indent=4)
            print(f"[seg]    → {p}")

            seg_np = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
            seg_full = GetFullSizDetectionMap_nifti(seg_np, crop_coords, itk_img)
            p = seg_dir / f"{casename}_segmentation_fullsize.nii.gz"
            sitk.WriteImage(seg_full, str(p))
            print(f"[seg]    → {p}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete.  Outputs in: {output_dir}")
    enabled = []
    if do_radiomics:     enabled.append("radiomics/")
    if do_cam:           enabled.append("cam/")
    if do_segmentation:  enabled.append("segmentation/")
    print(f"  Enabled: {', '.join(enabled)}")
    print(f"{'='*60}")


# ══════════════════════════ CLI ═══════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Unified PDAC pipeline: radiomics + CAM + segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # required
    p.add_argument('--input_image', '-i', required=True,
                   help='CT image (.nii.gz or .mha)')
    p.add_argument('--output_dir', '-o', required=True,
                   help='Root output directory')

    # optional mask
    p.add_argument('--pancreas_mask', '-m', default=None,
                   help='Pancreas mask; omit to auto-generate via lowres model')

    # output toggles
    p.add_argument('--radiomics', action='store_true',
                   help='Save resampled voxel feature maps + global features')
    p.add_argument('--cam', action='store_true',
                   help='Generate GradCAM++ and Input Gradient CAMs')
    p.add_argument('--segmentation', action='store_true',
                   help='Run nnUNet fullres segmentation')

    # model paths
    p.add_argument('--model_dir', default=str(SRC_DIR / "nnUNet_results"),
                   help='Root nnUNet_results/ dir')
    p.add_argument('--voxel_config', default=str(SRC_DIR / "PANORAMA_voxel.json"),
                   help='Voxel radiomics JSON config')

    # CAM options
    p.add_argument('--cam_types', nargs='+', default=['gradcam++', 'input_gradient'],
                   choices=['gradcam', 'gradcam++', 'input_gradient'])
    p.add_argument('--cam_target_class', type=int, default=1,
                   help='Target class for CAM (1=tumor)')
    p.add_argument('--cam_folds', nargs='+', type=int, default=[0],
                   help='Model folds for CAM forward pass')
    p.add_argument('--cam_device', default='cuda', choices=['cuda', 'cpu'])

    # segmentation options
    p.add_argument('--seg_trainer', default='nnUNetTrainer_Loss_CE_checkpoints')
    p.add_argument('--seg_checkpoint', default='checkpoint_best.pth')
    p.add_argument('--seg_folds', default='0,1,2,3,4',
                   help='Comma-separated folds for segmentation')

    args = p.parse_args()

    if not (args.radiomics or args.cam or args.segmentation):
        p.error("At least one of --radiomics, --cam, --segmentation is required.")

    run_pipeline(
        input_image=args.input_image,
        output_dir=args.output_dir,
        pancreas_mask=args.pancreas_mask,
        do_radiomics=args.radiomics,
        do_cam=args.cam,
        do_segmentation=args.segmentation,
        model_dir=args.model_dir,
        voxel_config=args.voxel_config,
        cam_types=args.cam_types,
        cam_target_class=args.cam_target_class,
        cam_folds=args.cam_folds,
        cam_device=args.cam_device,
        seg_trainer=args.seg_trainer,
        seg_checkpoint=args.seg_checkpoint,
        seg_folds=args.seg_folds,
    )


if __name__ == '__main__':
    main()
