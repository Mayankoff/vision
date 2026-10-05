<#
.SYNOPSIS
  Create the rPPG-Toolbox conda environment (Python 3.8 + PyTorch 2.1.2) on Windows.
.DESCRIPTION
  Needs Miniconda/Anaconda ("conda" on PATH; use the Anaconda PowerShell Prompt if it is not).
  mamba-ssm and causal-conv1d are skipped: they only build on Linux and are needed only by
  PhysMamba, which is not in our plan (scripts\toolbox\stubs provides a stand-in).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1          # NVIDIA GPU (CUDA 12.1)
  powershell -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1 -Cpu     # no GPU: classical methods / inference only
#>
param([string]$EnvName = "rppg-toolbox", [switch]$Cpu)
. (Join-Path $PSScriptRoot "common.ps1")

if (-not (Test-Path "external/rPPG-Toolbox/main.py")) { Invoke-Checked "git" @("submodule", "update", "--init") }

$envs = (& conda env list) -join "`n"
if ($envs -notmatch "(?m)^$([regex]::Escape($EnvName))\s") {
    Invoke-Checked "conda" @("create", "-y", "-n", $EnvName, "python=3.8")
}
$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu121" }
$run = @("run", "--no-capture-output", "-n", $EnvName, "python", "-m", "pip", "install")
Invoke-Checked "conda" ($run + @("torch==2.1.2", "torchvision==0.16.2", "--index-url", $index))

$req = Join-Path $env:TEMP "rppg_toolbox_requirements.txt"
Get-Content "external/rPPG-Toolbox/requirements.txt" | Where-Object { $_ -notmatch "^(mamba-ssm|causal-conv1d)" } | Set-Content $req
Invoke-Checked "conda" ($run + @("-r", $req))
Invoke-Checked "conda" @("run", "--no-capture-output", "-n", $EnvName, "python", "-c",
    "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())")
Write-Host "`nDone. Toolbox environment: $EnvName" -ForegroundColor Green
