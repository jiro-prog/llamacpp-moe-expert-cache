# Run the chat benchmark once per configuration: start qwen-run.ps1 with the given arguments, wait for /health,
# run chatbench.ps1, stop the server. ASCII only. Results: bench\<tag>-*.jsonl and one line each in bench\ab-summary.txt
#   .\abrun.ps1 -ConfigFile bench\ab-configs.txt     one "tag=qwen-run arguments" per line ('#' lines are skipped)
param([string]$ConfigFile = 'C:\llama-qwen\bench\ab-configs.txt', [int]$MaxTokens = 128)
$ErrorActionPreference = 'Continue'
$summary = 'C:\llama-qwen\bench\ab-summary.txt'
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
    if (-not $ok) {
        "{0:HH:mm:ss} {1}: server did not come up (runner exited={2})" -f (Get-Date), $tag, $runner.HasExited | Out-File $summary -Append -Encoding ascii
        & C:\llama-qwen\qwen-run.ps1 -Stop | Out-Null
        continue
    }
    $lines = & C:\llama-qwen\bench\chatbench.ps1 -Tag $tag -MaxTokens $MaxTokens 2>&1 | Out-String
    $lines -split "`n" | Where-Object { $_ -match 't/s' } | ForEach-Object { "  {0}: {1}" -f $tag, $_.Trim() } | Out-File $summary -Append -Encoding ascii
    & C:\llama-qwen\qwen-run.ps1 -Stop | Out-Null
    Start-Sleep 5
}
"{0:HH:mm:ss} ALL DONE" -f (Get-Date) | Out-File $summary -Append -Encoding ascii
