# Long-prompt A/B: one longbench.ps1 run per configuration (same file format as ab-configs.txt). ASCII only.
param([string]$ConfigFile = 'C:\llama-qwen\bench\long-configs.txt', [int]$Chars = 6000, [Parameter(Mandatory)][string]$Source)
$ErrorActionPreference = 'Continue'
$summary = 'C:\llama-qwen\bench\long-summary.txt'
$Configs = @(Get-Content $ConfigFile | Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith('#') })
foreach ($c in $Configs) {
    $tag, $args_ = $c -split '=', 2
    "{0:HH:mm:ss} start {1}: {2}" -f (Get-Date), $tag, $args_ | Out-File $summary -Append -Encoding ascii
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'C:\llama-qwen\qwen-run.ps1') + ($args_ -split ' ' | Where-Object { $_ })
    $runner = Start-Process powershell -ArgumentList $argList -WindowStyle Hidden -PassThru
    $t = Get-Date; $ok = $false
    do {
        Start-Sleep 3
        $ok = try { (Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 2).status -eq 'ok' } catch { $false }
    } until ($ok -or $runner.HasExited -or ((Get-Date) - $t).TotalSeconds -gt 300)
    if ($ok) {
        $line = & C:\llama-qwen\bench\longbench.ps1 -Tag $tag -Chars $Chars -Source $Source 2>&1 | Out-String
        "  {0}: {1}" -f $tag, $line.Trim() | Out-File $summary -Append -Encoding ascii
    } else {
        "  {0}: server did not come up" -f $tag | Out-File $summary -Append -Encoding ascii
    }
    & C:\llama-qwen\qwen-run.ps1 -Stop | Out-Null
    Start-Sleep 5
}
"{0:HH:mm:ss} ALL DONE" -f (Get-Date) | Out-File $summary -Append -Encoding ascii
