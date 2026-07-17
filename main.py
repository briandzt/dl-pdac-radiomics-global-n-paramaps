"""Convenience local entrypoint.

This mirrors the simple ``python main.py -i ... -o ...`` workflow used by
similar PANORAMA repositories while delegating to the canonical implementation
in ``src/process.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.process import PDACDetectionContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PANORAMA PDAC inference.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing input CT images.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory where outputs will be saved.")
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Local workspace for nnUNet_raw, nnUNet_preprocessed, and staged inputs/outputs.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to nnUNet_results. Defaults to <workspace>/nnUNet_results if present, otherwise src/nnUNet_results.",
    )
    parser.add_argument("--image-ext", default=".nii.gz", help="Input suffix to scan for, for example .nii.gz or .mha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    workspace = Path(args.workspace).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else workspace / "nnUNet_results"
    if not model_dir.exists():
        model_dir = repo_root / "src" / "nnUNet_results"

    PDACDetectionContainer(
        nnunet_base=workspace,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=str(model_dir),
        image_ext=args.image_ext,
    ).process()


if __name__ == "__main__":
    main()
