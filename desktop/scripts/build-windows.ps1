param(
  [switch]$NoBundle
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Require-Command {
  param([string]$Name)

  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Missing required command '$Name'. Install it and run this script again."
  }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$iconsDir = Join-Path $projectRoot "src-tauri/icons"
$svgIcon = Join-Path $iconsDir "martinus.svg"
$icoIcon = Join-Path $iconsDir "icon.ico"
$bundleArgs = @("--bundles", "msi,nsis")

$isWin = ($env:OS -eq "Windows_NT") -or ((Test-Path Variable:IsWindows) -and $IsWindows)
if (-not $isWin) {
  throw "This script is intended to run on Windows."
}

Write-Host "Martinus Windows build helper"
Write-Host "Project root: $projectRoot"

Require-Command "node"
Require-Command "npm"
Require-Command "cargo"
Require-Command "rustup"
Require-Command "npx"

Push-Location $projectRoot
try {
  if (-not (Test-Path (Join-Path $projectRoot "node_modules"))) {
    Write-Host "Installing desktop dependencies..."
    npm install
    if ($LASTEXITCODE -ne 0) {
      throw "npm install failed."
    }
  }

  if (-not (Test-Path $icoIcon)) {
    if (-not (Test-Path $svgIcon)) {
      throw "Cannot generate Windows icon because '$svgIcon' is missing."
    }

    Write-Host "Generating Windows icon assets..."
    npx tauri icon $svgIcon --output $iconsDir
    if ($LASTEXITCODE -ne 0) {
      throw "Icon generation failed."
    }
  }

  Write-Host "Ensuring the Windows Rust target is installed..."
  rustup target add x86_64-pc-windows-msvc
  if ($LASTEXITCODE -ne 0) {
    throw "rustup target add x86_64-pc-windows-msvc failed."
  }

  if ($NoBundle) {
    Write-Host "Running an unbundled Windows-targeted desktop build..."
    npm run tauri:build -- --target x86_64-pc-windows-msvc --no-bundle
  } else {
    Write-Host "Running the Windows installer build..."
    npm run tauri:build -- --target x86_64-pc-windows-msvc @bundleArgs
  }

  if ($LASTEXITCODE -ne 0) {
    throw "Windows build failed."
  }

  Write-Host ""
  Write-Host "Build completed."
  Write-Host "Artifacts: $projectRoot/src-tauri/target/x86_64-pc-windows-msvc/release/bundle"
}
finally {
  Pop-Location
}
