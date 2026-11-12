[CmdletBinding()]
param(
    [string] $PythonPath = 'python',
    [string] $OutputDirectory
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') {
    throw 'Build the Windows executable on Windows.'
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot 'build/annotator'
$entryPoint = Join-Path $repoRoot 'annotator/__main__.py'
$icon = Join-Path $repoRoot 'annotator/floor_contact.ico'
$requirements = Join-Path $PSScriptRoot 'requirements-annotator-build.txt'
$environment = Join-Path $buildRoot '.venv'
$buildPython = Join-Path $environment 'Scripts/python.exe'

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $buildRoot 'dist'
}
$OutputDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)

if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
    & $PythonPath -m venv $environment
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the build environment. Install 64-bit Python 3.12 or pass -PythonPath.'
    }
}

& $buildPython -c 'import sys, sysconfig; sys.exit(0 if sys.version_info[:2] == (3, 12) and sysconfig.get_platform() == ''win-amd64'' else 1)'
if ($LASTEXITCODE -ne 0) {
    throw "The build environment at $environment must use 64-bit Python 3.12."
}
& $buildPython -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install the build requirements.'
}

$packagingOptions = @(
    '--noconfirm', '--onefile', '--windowed', '--noupx'
    '--name', 'FloorContactLabeler'
    '--icon', $icon
    '--add-data', "$icon;annotator"
    '--paths', $repoRoot
    '--workpath', (Join-Path $buildRoot 'work')
    '--specpath', $buildRoot
    '--distpath', $OutputDirectory
    $entryPoint
)
& $buildPython -m PyInstaller @packagingOptions
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller could not build the annotator.'
}

$executable = Join-Path $OutputDirectory 'FloorContactLabeler.exe'
Get-Item -LiteralPath $executable | Select-Object FullName, Length
Get-FileHash -LiteralPath $executable -Algorithm SHA256
