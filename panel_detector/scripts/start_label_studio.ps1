# Launch Label Studio with Local Files Storage rooted at the repo.
# Usage:  pwsh panel_detector/scripts/start_label_studio.ps1
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = 'true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $repoRoot
$env:LABEL_STUDIO_BASE_DATA_DIR = Join-Path $repoRoot 'panel_detector/labelstudio/.lsdata'
Write-Host "repo root         : $repoRoot"
Write-Host "LS data dir       : $env:LABEL_STUDIO_BASE_DATA_DIR"
Write-Host "Local Files root  : $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
Write-Host ""
Write-Host "Open http://localhost:8080 in your browser when it boots."
Write-Host "After login, follow panel_detector/labelstudio/README.md for"
Write-Host "the 3-click project setup."
label-studio start
