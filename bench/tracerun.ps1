# Record a routing trace over the 10-question Japanese check, detached (Start-Process). ASCII only.
# Progress: bench\<Tag>-status.txt (last line "DONE" when finished). Trace: bench\<Tag>-trace.txt
param([string]$Tag = "trace-$(Get-Date -Format 'MMdd-HHmm')", [string]$RunArgs = '-Stats -Timing')
$status = "C:\llama-qwen\bench\$Tag-status.txt"
function S($s) { "$(Get-Date -Format 'HH:mm:ss') $s" | Out-File -FilePath $status -Append -Encoding ascii }
S "start $Tag ($RunArgs)"
$argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'C:\llama-qwen\qwen-run.ps1') + ($RunArgs -split ' ' | Where-Object { $_ }) +
    @('-Trace', "C:\llama-qwen\bench\$Tag-trace.txt")
$runner = Start-Process powershell -ArgumentList $argList -WindowStyle Hidden -PassThru
$t = Get-Date; $ok = $false
do {
    Start-Sleep 3
    $ok = try { (Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 2).status -eq 'ok' } catch { $false }
} until ($ok -or $runner.HasExited -or ((Get-Date) - $t).TotalSeconds -gt 300)
if ($ok) {
    S 'server up'
    try { & C:\llama-qwen\bench\jaquality.ps1 -Tag "jaq-$Tag" | ForEach-Object { S "jaq $_" } } catch { S "ERROR: $_" }
} else { S 'server did not come up' }
& C:\llama-qwen\qwen-run.ps1 -Stop | Out-Null
S 'DONE'
