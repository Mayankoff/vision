<#
.SYNOPSIS
  Phase 1 check: cache patch traces for a dataset, then score every variant (single region ... full method).
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase1.ps1 -Dataset ubfc
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase1.ps1 -Dataset pure -Workers 6
#>
param(
    [ValidateSet("ubfc", "pure", "synthetic")][string]$Dataset = "ubfc",
    [string]$Root = "",
    [int]$Workers = 4
)
. (Join-Path $PSScriptRoot "common.ps1")
$py = Get-OurPython
if (-not $Root) { $Root = $DatasetDirs[$Dataset] }
if (-not (Test-Path $Root)) { throw "Dataset folder not found: $Root (see docs\DATA.md)" }
Invoke-Checked $py @("-m", "afscreen.cli", "extract", $Dataset, $Root, "--out", "cache/traces/$Dataset", "--workers", "$Workers")
Invoke-Checked $py @("-m", "afscreen.cli", "evaluate", "cache/traces/$Dataset", "--out", "results/phase1/$Dataset")
Write-Host "`nResults: results\phase1\$Dataset\summary.md" -ForegroundColor Green
