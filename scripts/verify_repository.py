"""CPU-safe repository sanity checks.

This script intentionally avoids importing torch, SimpleITK, nnUNet, or other
heavy dependencies. It verifies that the publishable repository has the files
and checkpoint placeholders needed before a Docker/GPU inference run.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1 = (
    REPO_ROOT
    / "src"
    / "nnUNet_results"
    / "Dataset001_LR"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
STAGE2 = (
    REPO_ROOT
    / "src"
    / "nnUNet_results"
    / "Dataset002_stage2"
    / "nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres"
)


REQUIRED_FILES = [
    "Dockerfile",
    "main.py",
    "requirements.txt",
    "src/process.py",
    "src/process_local.py",
    "src/data_utils.py",
    "src/voxel_radiomics.py",
    "src/PANORAMA_voxel.json",
    "src/nnunetv2_global_rad/pyproject.toml",
    "src/dynamic-network-architectures_global_rad/dynamic-network-architectures/setup.py",
    "src/pyradiomics-3.1.0-Zengtian/setup.py",
    "src/pytorchradiomics-main/pyproject.toml",
    "workspace/nnUNet_raw/.gitkeep",
    "workspace/nnUNet_preprocessed/.gitkeep",
    "workspace/nnUNet_results/.gitkeep",
    "workspace/test_example/input/.gitkeep",
    "workspace/test_example/output/.gitkeep",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def check_required_files() -> None:
    for rel in REQUIRED_FILES:
        require((REPO_ROOT / rel).exists(), f"missing {rel}")


def check_nnunet_metadata(model_dir: Path) -> None:
    for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
        path = model_dir / name
        require(path.exists(), f"missing {path.relative_to(REPO_ROOT)}")
    for name in ("dataset.json", "plans.json"):
        with (model_dir / name).open("r", encoding="utf-8") as f:
            json.load(f)
    for fold in range(5):
        fold_dir = model_dir / f"fold_{fold}"
        require(fold_dir.exists(), f"missing {fold_dir.relative_to(REPO_ROOT)}")


def report_weights_in_tree() -> None:
    weights = [p for p in REPO_ROOT.rglob("*.pth") if ".git" not in p.parts]
    if weights:
        print(f"NOTE: found {len(weights)} local .pth checkpoint file(s). Ensure they remain untracked.")


def report_checkpoint_readiness() -> None:
    for label, model_dir in (("Stage-1 Dataset001_LR", STAGE1), ("Stage-2 Dataset002_stage2", STAGE2)):
        ready = []
        missing = []
        for fold in range(5):
            checkpoint = model_dir / f"fold_{fold}" / "checkpoint_best.pth"
            if checkpoint.exists():
                ready.append(fold)
            else:
                missing.append(fold)
        if missing:
            print(f"NOTE: {label} missing checkpoint_best.pth for folds {missing}.")
        else:
            print(f"OK: {label} has checkpoint_best.pth for folds {ready}.")


def main() -> None:
    check_required_files()
    check_nnunet_metadata(STAGE1)
    check_nnunet_metadata(STAGE2)
    report_weights_in_tree()
    report_checkpoint_readiness()
    print("OK: repository structure is ready for CPU-side review.")


if __name__ == "__main__":
    main()
