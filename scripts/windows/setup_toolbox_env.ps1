<#
.SYNOPSIS
  Create the rPPG-Toolbox environment (Python 3.8 + PyTorch 2.1.2) in .venv-toolbox.
.DESCRIPTION
  Uses uv (installed into our .venv), which downloads Python 3.8 itself - no conda needed.
  Run scripts\windows\setup.ps1 first. Pass -Conda to use a conda env instead.
  mamba-ssm and causal-conv1d are skipped: they only build on Linux and are needed only by
  PhysMamba, which is not in our plan (scripts\toolbox\stubs provides a stand-in).
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1        # NVIDIA GPU (CUDA 12.1)
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup_toolbox_env.ps1 -Cpu   # no NVIDIA GPU
#>
param([switch]$Cpu, [switch]$Conda, [string]$EnvName = "rppg-toolbox")
. (Join-Path $PSScriptRoot "common.ps1")

if (-not (Test-Path "external/rPPG-Toolbox/main.py")) { Invoke-Checked "git" @("submodule", "update", "--init") }
$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu121" }
$req = Join-Path ([System.IO.Path]::GetTempPath()) "rppg_toolbox_requirements.txt"
Get-Content "external/rPPG-Toolbox/requirements.txt" | Where-Object { $_ -notmatch "^(mamba-ssm|causal-conv1d)" } | Set-Content $req

if ($Conda) {
    $envs = (& conda env list) -join "`n"
    if ($envs -notmatch "(?m)^$([regex]::Escape($EnvName))\s") { Invoke-Checked "conda" @("create", "-y", "-n", $EnvName, "python=3.8") }
    $run = @("run", "--no-capture-output", "-n", $EnvName, "python", "-m", "pip", "install")
    Invoke-Checked "conda" ($run + @("torch==2.1.2", "torchvision==0.16.2", "--index-url", $index))
    Invoke-Checked "conda" ($run + @("-r", $req))
    $check = @("conda", @("run", "--no-capture-output", "-n", $EnvName, "python"))
} else {
    $py = Get-OurPython
    Invoke-Checked $py @("-m", "pip", "install", "--quiet", "uv")
    $uv = Get-ChildItem (Split-Path $py) -Filter "uv*" | Where-Object { $_.BaseName -eq "uv" } | Select-Object -First 1
    if (-not $uv) { throw "uv was not installed into .venv" }
    if (-not (Test-Path ".venv-toolbox")) { Invoke-Checked $uv.FullName @("venv", "--python", "3.8", ".venv-toolbox") }
    $tpy = Get-ToolboxVenvPython
    Invoke-Checked $uv.FullName @("pip", "install", "--python", $tpy, "torch==2.1.2", "torchvision==0.16.2", "--index-url", $index)
    Invoke-Checked $uv.FullName @("pip", "install", "--python", $tpy, "-r", $req)
    $check = @($tpy, @())
}
Invoke-Checked $check[0] ($check[1] + @("-c", "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"))
Write-Host "`nDone. Toolbox environment ready." -ForegroundColor Green
