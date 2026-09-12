param([switch]$SkipModels, [switch]$WithWeb)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) { throw 'Install 64-bit Python 3.12 and its py launcher first.' }
if (-not (Test-Path -LiteralPath '.venv-model/Scripts/python.exe')) {
    & py.exe -3.12 -m venv .venv-model
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$Python = Join-Path $ProjectRoot '.venv-model/Scripts/python.exe'
function Run-PythonStep([string[]]$StepArguments) {
    & $Python @StepArguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed with exit code $LASTEXITCODE" }
}
Run-PythonStep @('-m','pip','install','--upgrade','pip')
Run-PythonStep @('-m','pip','install','-e','.')
Run-PythonStep @('-m','pip','install','-r','requirements-models-onnx.txt')
Run-PythonStep @('-m','pip','install','--no-deps','rtmlib==0.0.16')
Run-PythonStep @('-m','pip','install','-r','requirements-models-cuda.txt','--index-url','https://download.pytorch.org/whl/cu130')
Run-PythonStep @('-m','pip','install','-r','requirements-models-app.txt')
Run-PythonStep @('-m','pip','install','--no-deps','ultralytics==8.4.116')
if (-not $SkipModels) { Run-PythonStep @('scripts/download_models.py','--include-bst') }
if ($WithWeb) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 22.13 or newer for the optional UI.' }
    Push-Location demo
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    } finally { Pop-Location }
}
Write-Host 'Setup complete. Run Python tests, then scripts/verify_portable.py --full after downloading models.'
