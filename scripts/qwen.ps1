# Qwen3.8-Flash-Next server control. ASCII only.
#   .\qwen.ps1 start          start for LAN + Tailscale (http://<this PC>:8090, API key in api-key.txt)
#   .\qwen.ps1 start -Local   start for this PC only (http://127.0.0.1:8091, no API key)
#   .\qwen.ps1 stop
#   .\qwen.ps1 status
param(
    [Parameter(Position = 0)][ValidateSet('start', 'stop', 'status')][string]$Command = 'status',
    [switch]$Local,
    [switch]$Think
)
$root = 'C:\llama-qwen'

# a server has dozens of threads; an exe stuck at exit in the GPU driver (seen 2026-10-03, unkillable until a reboot)
# keeps only one and must not count as running
function Get-Qwen { Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root*" -and $_.Threads.Count -gt 1 } }
function Test-Health($port) { try { (Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2).status -eq 'ok' } catch { $false } }

switch ($Command) {
    'stop' {
        $p = Get-Qwen
        if ($p) { $p | Stop-Process -Force; "stopped pid $($p.Id -join ', ')" } else { 'not running' }
        # the watchdog loop in qwen-run.ps1 exits by itself once the server is gone
    }
    'start' {
        if (Get-Qwen) { 'already running (qwen.ps1 stop first)'; return }
        $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'qwen-run.ps1'))
        if (-not $Local) { $a += '-Lan' }
        if ($Think) { $a += '-Think' }
        Start-Process powershell -ArgumentList $a -WindowStyle Hidden | Out-Null
        $port = if ($Local) { 8091 } else { 8090 }
        $t = Get-Date
        while (-not (Test-Health $port) -and ((Get-Date) - $t).TotalSeconds -lt 120) { Start-Sleep 2 }
        if (-not (Test-Health $port)) { 'did not come up in 120 s - see logs\server-*.log'; return }
        "ready in {0:N0} s" -f ((Get-Date) - $t).TotalSeconds
        if ($Local) { 'open http://127.0.0.1:8091 on this PC' }
        else {
            # this PC's addresses (LAN, and the VPN adapter if any)
            Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
                ForEach-Object { 'URL:       http://{0}:8090  ({1})' -f $_.IPAddress, $_.InterfaceAlias }
            'API key:   ' + (Get-Content (Join-Path $root 'api-key.txt'))
        }
    }
    'status' {
        $p = Get-Qwen
        if (-not $p) { 'STOPPED'; return }
        $avail = (New-Object System.Diagnostics.PerformanceCounter('Memory', 'Available MBytes', '', $true)).RawValue
        "RUNNING pid {0}, private {1:N0} MB, working set {2:N0} MB | available RAM {3:N0} MB" -f $p.Id, ($p.PrivateMemorySize64 / 1MB), ($p.WorkingSet64 / 1MB), $avail
        $log = Get-ChildItem (Join-Path $root 'logs\server-*.log') -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($log) {
            $l = Select-String -Path $log.FullName -Pattern 'listening on' -SimpleMatch | Select-Object -Last 1
            if ($l) { 'listening: ' + ($l.Line -replace '^.*listening on ', '') }
            $tm = Select-String -Path $log.FullName -Pattern '        eval time' -SimpleMatch | Select-Object -Last 1
            if ($tm) { 'last reply: ' + ($tm.Line -replace '^.*eval time =\s*', '') }
        }
    }
}
