$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DemoRoot = Join-Path $ProjectRoot "demo"
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Npm) {
    throw "npm.cmd was not found. Install Node.js 22.13 or newer."
}
if (-not (Test-Path -LiteralPath (Join-Path $DemoRoot "node_modules"))) {
    Push-Location $DemoRoot
    try { & npm.cmd ci }
    finally { Pop-Location }
}
Set-Location -LiteralPath $DemoRoot
Write-Host "Gold UI: http://127.0.0.1:3000/gold"
& npm.cmd run dev -- --host 127.0.0.1 --port 3000
