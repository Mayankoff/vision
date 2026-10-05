<#
.SYNOPSIS
  Phase 1: our classical pipeline on one dataset (face landmarks -> traces -> GREEN/ICA/CHROM/POS -> scores).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\run_phase1.ps1 -Dataset ubfc
#>
param(
    [ValidateSet("ubfc", "pure", "mmpd")][string]$Dataset = "ubfc",
    [string]$Root = "",
    [int]$Workers = 4,
    [switch]$DebugVideos
)
. (Join-Path $PSScriptRoot "common.ps1")
$py = Get-OurPython
if (-not $Root) { $Root = $DatasetDirs[$Dataset] }
if (-not (Test-Path $Root)) { throw "Dataset folder not found: $Root (see README, 'Data')" }

$extract = @("scripts/extract_traces.py", $Dataset, $Root, "--out", "cache/traces/$Dataset", "--workers", "$Workers")
if ($DebugVideos) { $extract += @("--debug-videos", "cache/debug/$Dataset") }
Invoke-Checked $py $extract
Invoke-Checked $py @("scripts/run_classical.py", "cache/traces/$Dataset", "--out", "results/classical/$Dataset", "--window", "10")
Invoke-Checked $py @("scripts/run_classical.py", "cache/traces/$Dataset", "--split", "splits/$Dataset.json", "--subset", "test",
    "--out", "results/classical/${Dataset}_test")
Write-Host "`nResults: results\classical\${Dataset}_summary.csv" -ForegroundColor Green
