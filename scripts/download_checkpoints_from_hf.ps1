param(
    [string]$RepoId = "briandzt/radiomics_nnUNet",

    [string]$RepoRoot,

    [string]$HfExe = "$env:USERPROFILE\anaconda3\envs\hf-cli\Scripts\hf.exe"
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not (Test-Path -LiteralPath $HfExe)) {
    throw "Cannot find hf executable at $HfExe. Activate/install Hugging Face CLI first."
}

$resultsRoot = Join-Path $RepoRoot "src\nnUNet_results"

& $HfExe download $RepoId `
    --repo-type model `
    --local-dir $resultsRoot `
    --include "Dataset001_LR/**" `
    --include "Dataset002_stage2/**"

Write-Host "Downloaded model files to $resultsRoot"
Write-Host "Run: python scripts\verify_repository.py"
