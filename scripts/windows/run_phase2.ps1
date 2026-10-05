<#
.SYNOPSIS
  Phase 2: rPPG-Toolbox classical methods vs ours on every dataset present, plus the ROI ablation report.
.DESCRIPTION
  For each dataset folder that exists under data\: runs Phase 1 if its traces are missing, runs the toolbox's
  methods (toolbox environment), scores them with our protocol, then writes results\phase2\REPORT.md.
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase2.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_phase2.ps1 -Datasets ubfc -ToolboxPython C:\miniconda3\envs\rppg-toolbox\python.exe
#>
param(
    [string[]]$Datasets = @("ubfc", "pure", "mmpd"),
    [string]$ToolboxEnv = "rppg-toolbox",
    [string]$ToolboxPython = "",
    [int]$Workers = 4
)
. (Join-Path $PSScriptRoot "common.ps1")
# With "powershell -File", "-Datasets ubfc,pure" arrives as one string; split it ourselves.
$Datasets = @($Datasets | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
foreach ($ds in $Datasets) { if (-not $DatasetDirs.ContainsKey($ds)) { throw "unknown dataset '$ds' (use ubfc, pure, mmpd)" } }
$py = Get-OurPython
$csvs = @()
foreach ($ds in $Datasets) {
    $root = $DatasetDirs[$ds]
    if (-not (Test-Path $root)) { Write-Host "skip ${ds}: $root not found" -ForegroundColor Yellow; continue }
    $tb = $ToolboxNames[$ds]

    if (-not (Test-Path "results/classical/${ds}_per_video.csv")) {
        & (Join-Path $PSScriptRoot "run_phase1.ps1") -Dataset $ds -Workers $Workers
    }
    $cfg = "configs/toolbox/${tb}_UNSUPERVISED.yaml"
    $cached = "cache/toolbox/preprocessed/${tb}_HC72_raw"
    if (Test-Path $cached) {
        # Face crops already cached: tell the toolbox not to redo them.
        $tmp = "cache/toolbox/${tb}_UNSUPERVISED_cached.yaml"
        (Get-Content $cfg) -replace "DO_PREPROCESS: True", "DO_PREPROCESS: False" | Set-Content $tmp
        $cfg = $tmp
    }
    Invoke-Toolbox $ToolboxPython $ToolboxEnv @("scripts/toolbox/dump_unsupervised.py", "--config_file", $cfg)
    Invoke-Checked $py @("scripts/score_predictions.py", "cache/toolbox/predictions/$tb", "--out", "results/toolbox/$ds")
    $csvs += @("results/classical/${ds}_per_video.csv", "results/toolbox/${ds}_per_video.csv")
}
if ($csvs.Count -eq 0) { throw "No datasets found under data\ (see README, 'Data')." }
Invoke-Checked $py (@("scripts/phase2_report.py") + $csvs + @("--out", "results/phase2"))
Write-Host "`nReport: results\phase2\REPORT.md" -ForegroundColor Green
