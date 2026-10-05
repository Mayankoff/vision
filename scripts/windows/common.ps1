# Shared helpers for the Windows scripts (Windows PowerShell 5.1 compatible).
$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $Repo

$DatasetDirs = @{ ubfc = "data/UBFC-rPPG"; pure = "data/PURE"; mmpd = "data/MMPD" }
$ToolboxNames = @{ ubfc = "UBFC-rPPG"; pure = "PURE"; mmpd = "MMPD" }

function Invoke-Checked {
    # Run a native command and stop if it fails (PowerShell 5.1 does not do this by itself).
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "command failed with exit code ${LASTEXITCODE}: $Exe $($Arguments -join ' ')" }
}

function Get-OurPython {
    $py = Join-Path $Repo ".venv/Scripts/python.exe"
    if (-not (Test-Path $py)) { $py = Join-Path $Repo ".venv/bin/python" }  # lets the scripts run under pwsh on Linux too
    if (-not (Test-Path $py)) { throw "Our environment is missing. Run scripts\windows\setup.ps1 first." }
    return $py
}

function Invoke-Toolbox {
    # Run a Python script in the rPPG-Toolbox environment: either a python.exe path, or a conda env name.
    param([string]$ToolboxPython, [string]$ToolboxEnv, [string[]]$Arguments)
    if ($ToolboxPython) {
        Invoke-Checked $ToolboxPython $Arguments
    } else {
        Invoke-Checked "conda" (@("run", "--no-capture-output", "-n", $ToolboxEnv, "python") + $Arguments)
    }
}
