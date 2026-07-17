#!/usr/bin/env python
"""
Compute voxel volume of label 1 for all segmentation files in a folder.

Usage:
    python compute_volumes.py --input_dir /path/to/labels --output volumes.csv
    python compute_volumes.py -i /path/to/labels -o volumes.csv --label 2
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def compute_label_volume(filepath, label=1):
    """Return volume in mm³ for the given label value."""
    img = sitk.ReadImage(str(filepath))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (x, y, z) in mm
    voxel_vol = float(np.prod(spacing))
    count = int(np.sum(arr == label))
    return count, count * voxel_vol


def main():
    p = argparse.ArgumentParser(description="Compute label-1 voxel volume for a folder of segmentations")
    p.add_argument("--input_dir", "-i", required=True, help="Folder containing .nii.gz / .mha label files")
    p.add_argument("--output", "-o", required=True, help="Output CSV path")
    p.add_argument("--label", "-l", type=int, default=1, help="Label value to measure (default: 1)")
    p.add_argument("--ext", default=".nii.gz", help="File extension filter (default: .nii.gz)")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(f"*{args.ext}"))
    if not files:
        files = sorted(input_dir.glob("*.mha"))
    if not files:
        print(f"No segmentation files found in {input_dir}")
        return

    rows = []
    for f in files:
        voxel_count, volume_mm3 = compute_label_volume(f, args.label)
        volume_ml = volume_mm3 / 1000.0
        rows.append({"filename": f.name, "voxel_count": voxel_count,
                      "volume_mm3": round(volume_mm3, 2), "volume_ml": round(volume_ml, 4)})
        print(f"{f.name}: {voxel_count} voxels, {volume_mm3:.2f} mm³, {volume_ml:.4f} mL")

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "voxel_count", "volume_mm3", "volume_ml"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} entries → {args.output}")


if __name__ == "__main__":
    main()
