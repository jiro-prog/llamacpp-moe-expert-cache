# One-time setup of the working directory (default C:\llama-qwen, the path the other scripts use). ASCII only.
#   1. llama.cpp b11361 source (commit a4cb4c61f) -> src\llama.cpp-a4cb4c61f, with patches\llama.cpp-b11361.patch applied
#   2. the b11361 Windows CUDA 12.4 release (for its ggml-cuda.dll and the CUDA runtime DLLs) -> bin\
#   3. build.cmd / noprefix.cmake and the run scripts copied next to them
# Then: build.cmd, tools\densecopy.exe (see README), qwen-run.ps1.
param([string]$Root = 'C:\llama-qwen')
$ErrorActionPreference = 'Stop'
$repo   = Split-Path $PSScriptRoot -Parent
$commit = 'a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd'
$rel    = 'https://github.com/ggml-org/llama.cpp/releases/download/b11361'
New-Item -ItemType Directory -Force (Join-Path $Root 'dl'), (Join-Path $Root 'src'), (Join-Path $Root 'bin'), (Join-Path $Root 'logs'), (Join-Path $Root 'bench') | Out-Null

function Get-File($url, $dst) {
    if (-not (Test-Path $dst)) { "download $url"; Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing }
}

$src = Join-Path $Root 'src\llama.cpp-a4cb4c61f'
if (-not (Test-Path $src)) {
    $zip = Join-Path $Root "dl\llama.cpp-$commit.zip"
    Get-File "https://github.com/ggml-org/llama.cpp/archive/$commit.zip" $zip
    Expand-Archive $zip (Join-Path $Root 'src') -Force
    Rename-Item (Join-Path $Root "src\llama.cpp-$commit") 'llama.cpp-a4cb4c61f'
    Push-Location $src
    try { git apply --whitespace=nowarn (Join-Path $repo 'patches\llama.cpp-b11361.patch'); if ($LASTEXITCODE) { throw 'git apply failed' } }
    finally { Pop-Location }
    'patched source: ' + $src
}

foreach ($z in @('llama-b11361-bin-win-cuda-12.4-x64.zip', 'cudart-llama-bin-win-cuda-12.4-x64.zip')) {
    $zip = Join-Path $Root "dl\$z"
    Get-File "$rel/$z" $zip
    Expand-Archive $zip (Join-Path $Root 'bin') -Force
}

Copy-Item (Join-Path $repo 'scripts\*') $Root -Force
Copy-Item (Join-Path $repo 'bench\*') (Join-Path $Root 'bench') -Force
'done. next: build.cmd (gcc/cmake/ninja on PATH or TOOLS_PATH), then the expert copy (README)'
