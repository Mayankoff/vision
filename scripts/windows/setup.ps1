<#
.SYNOPSIS
  Create our Python environment (.venv) and install the rppg package. Needs Python 3.11-3.13.
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#>
param([string]$Python = "")
. (Join-Path $PSScriptRoot "common.ps1")

if (-not $Python) {
    # Prefer the py launcher, which can pick a specific version.
    foreach ($v in "3.12", "3.11", "3.13") {
        try { & py "-$v" -c "import sys" 2>$null } catch { break }  # no py launcher installed
        if ($LASTEXITCODE -eq 0) { $Python = "py"; $PyArgs = @("-$v"); break }
    }
    if (-not $Python) { $Python = "python"; $PyArgs = @() }
} else { $PyArgs = @() }

$ver = & $Python @PyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.11") { throw "Python $ver found; 3.11, 3.12 or 3.13 is required (numpy 2.4 has no 3.10 build)." }
Write-Host "Using Python $ver"

if (-not (Test-Path ".venv")) { Invoke-Checked $Python ($PyArgs + @("-m", "venv", ".venv")) }
$py = Get-OurPython
Invoke-Checked $py @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $py @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $py @("-m", "pip", "install", "-e", ".")
Invoke-Checked $py @("-c", "from rppg.face.landmarker import ensure_model; print('model:', ensure_model())")
Invoke-Checked $py @("-m", "pytest", "-q")
Write-Host "`nDone. Activate with:  .venv\Scripts\Activate.ps1" -ForegroundColor Green
