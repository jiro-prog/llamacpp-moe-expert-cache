param([string]$Tag, [string]$RunArgs, [string]$Source, [string]$Out)
$argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'C:\llama-qwen\qwen-run.ps1') + ($RunArgs -split ' ' | Where-Object { $_ })
$p = Start-Process powershell -ArgumentList $argList -WindowStyle Hidden -PassThru
$t = Get-Date
do { Start-Sleep 3; $ok = try { (Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 2).status -eq 'ok' } catch { $false } } until ($ok -or ((Get-Date) - $t).TotalSeconds -gt 300)
# warm-up request so the cache is set up (it is created on the first pass)
$warm = @{ messages = @(@{ role = 'user'; content = 'hi' }); max_tokens = 4; temperature = 0 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8091/v1/chat/completions -Method Post -ContentType 'application/json' -Body $warm -TimeoutSec 600 | Out-Null
& C:\llama-qwen\bench\promptsweep.ps1 -Source $Source -Tag $Tag -Chars 100,200,400,800,1600,3200 -Out $Out
& C:\llama-qwen\qwen-run.ps1 -Stop | Out-Null
"$Tag DONE" | Out-File $Out -Append -Encoding ascii
