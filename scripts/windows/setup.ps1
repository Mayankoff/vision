<#
.SYNOPSIS
  Create the Python environment (.venv) and install the afscreen package. Needs Python 3.11-3.13.
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

if (Test-Path ".venv") {
    # An environment from the earlier version of the project has the old 'rppg' package; remove it.
    & (Get-OurPython) -m pip uninstall -y rppg 2>$null | Out-Null
} else {
    Invoke-Checked $Python ($PyArgs + @("-m", "venv", ".venv"))
}
$py = Get-OurPython
Invoke-Checked $py @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $py @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $py @("-m", "pip", "install", "-e", ".")
Invoke-Checked $py @("-c", "from afscreen.face.tracker import ensure_model; print('model:', ensure_model())")
Invoke-Checked $py @("-m", "pytest", "-q")
Write-Host "`nDone. Activate with:  .venv\Scripts\Activate.ps1" -ForegroundColor Green
