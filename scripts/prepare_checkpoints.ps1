param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointRoot,

    [string]$Stage1ModelDir,

    [string]$RepoRoot,

    [switch]$Link
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$resultsRoot = Join-Path $RepoRoot "src\nnUNet_results"
$stage1 = Join-Path $resultsRoot "Dataset001_LR\nnUNetTrainer__nnUNetPlans__3d_fullres"
$stage2 = Join-Path $resultsRoot "Dataset002_stage2\nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres"

function Copy-Or-LinkCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Force
    }

    if ($Link) {
        New-Item -ItemType HardLink -Path $Target -Target (Resolve-Path $Source).Path | Out-Null
    } else {
        Copy-Item -LiteralPath $Source -Destination $Target
    }
}

if ($Stage1ModelDir) {
    for ($fold = 0; $fold -lt 5; $fold++) {
        $source = Join-Path $Stage1ModelDir "fold_$fold\checkpoint_best.pth"
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Missing Stage-1 checkpoint: $source"
        }

        $targetDir = Join-Path $stage1 "fold_$fold"
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        $target = Join-Path $targetDir "checkpoint_best.pth"
        Copy-Or-LinkCheckpoint -Source $source -Target $target

        Write-Host "Prepared Stage-1 fold_$fold checkpoint_best.pth"
    }
} else {
    Write-Host "Stage-1 checkpoint source not provided; skipping Dataset001_LR weights."
}

for ($fold = 0; $fold -lt 5; $fold++) {
    $source = Join-Path $CheckpointRoot "fold$fold\checkpoint_1009.pth"
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing Stage-2 checkpoint: $source"
    }

    $targetDir = Join-Path $stage2 "fold_$fold"
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    $target = Join-Path $targetDir "checkpoint_best.pth"
    Copy-Or-LinkCheckpoint -Source $source -Target $target

    Write-Host "Prepared Stage-2 fold_$fold checkpoint_best.pth"
}

Write-Host "Done. Do not commit the generated .pth files."
