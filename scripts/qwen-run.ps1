# Qwen3.8-Flash-Next runner with a memory watchdog. ASCII only.
#   .\qwen-run.ps1                 start (127.0.0.1:8091) and watch until the server exits
#   .\qwen-run.ps1 -Stop           stop the server
# Memory is read with in-process performance counters (no WMI polling, see the 2026-10-01 BSOD).
# Defaults (A/B on 2026-10-03, bench\ab-summary / long-summary): own build + expert prefetch with 8 workers, no eviction,
# hard working-set cap 9 GB, ubatch 2048. Decode 1.5 -> 3.2 tok/s, a 3.5k-token prompt 1.0 -> 77 tok/s.
# 2026-10-05: + GPU keep-alive (-KeepAlive 300): the GPU no longer drops to P5 during decode, 3 questions 114 -> 88 s.
#             + expert cache (-CacheGB 7, 7.4 GB locked, misses read from experts-dense.bin): 88 -> 80 s.
#             (those numbers had -Stats lines written straight to stderr, which cost ~30 ms a token; now buffered)
#             + next-layer read-ahead (-Predict 8 -PredictWorkers 2) and cached-experts-first in MUL_MAT_ID:
#             3 questions 67 s, decode 5.7-7.2 tok/s; 10-question check 677 s, same answers as before.
#             Long prompts: 65 t/s (77 without the cache: the mapped part of the working set is 2 GB now, see -CacheMmapGB).
param(
    [int]$Ctx = 8192,
    [int]$Ubatch = 2048,           # prompts: one ubatch reads most experts once, so bigger is faster (512: 30 t/s, 2048: 77 t/s)
    [int]$Threads = 8,
    [string]$ListenHost = '127.0.0.1',
    [int]$Port = 8091,
    [string]$Ot = '\.ffn_(gate|up|down)_exps\.=CPU,per_layer_token_embd=CPU',
    [string]$Extra = '',
    [int]$MinCommitFreeMB = 1500,  # kill if system commit headroom drops below this
    [int]$MinAvailMB = 150,        # kill if Available RAM stays below this for 10 samples in a row
    [string]$Exe = 'C:\llama-qwen\build\bin\llama-server.exe',   # own build (build.cmd); release: C:\llama-qwen\bin\llama-server.exe
    [switch]$NoPrefetch,           # turn off the expert prefetch (own build: common\expert-prefetch.cpp; the release has none)
    [int]$Workers = 8,             # prefetch workers (8, 10, 12 measured the same)
    [int]$Evict = 0,               # drop expert slices this many layers back from the working set (0 = off: the working set
                                   # is the expert cache; evict 2 was 126 s vs 115 s on the 3-question bench)
    [int]$Touch = 1,               # workers touch the prefetched pages (fault them in off the compute threads)
    [switch]$Stats,                # one line per pass from the prefetcher
    [switch]$Timing,               # with -Stats: split each pass into CPU-MoE / fetch / GPU time
    [switch]$Diag,                 # a diag line per pass: pages already in the working set, disk reads, page faults, fault cost
    [string]$Trace = '',           # append the routing (experts per layer and pass) to this file
    [int]$KeepAlive = 300,         # GPU keep-alive: spin one warp this many us at a time while the CPU computes experts, so the
                                   # GPU stays in P2 during decode (0 = off; 2026-10-05: 3 questions 114 -> 88 s, GPU +35-45 W while generating)
    [double]$MaxWsGB = 9,          # hard working-set cap (0 = none): mapped pages above it go to the standby list.
                                   # 10 GB was 2% faster but left 1.5-2.8 GB of Available RAM (9 GB: 4.3 GB)
    [double]$CacheGB = 8.5,        # expert cache: decode reads experts from this much locked memory, filled by unbuffered reads of
                                   # -CacheFile (tools\densecopy.exe) instead of the mapped GGUF (0 = off). The cache then sets the
                                   # working set itself: cache + -CacheMmapGB (hard cap), so -MaxWsGB only applies until then.
                                   # 2026-10-05: 10-question check 1099 -> 847 s, same answers, Available RAM >= 3.9 GB
    [string]$CacheFile = 'C:\models\Qwen3.8-Flash-Next\UD-Q4_K_XL\experts-dense',
    [double]$CacheMmapGB = 0.6,    # working set left for the mapped file in decode (n-gram rows etc.); see -PrefillReleaseGB
    [double]$PrefillReleaseGB = 3.5, # cache given to the mapped side while a GPU prompt batch runs (taken back for decode;
                                   # 2026-10-06, 3551 tokens: 0 -> 49.5 t/s, 2.5 -> 64.6, 3.5 -> 65.5)
    [int]$Predict = 8,             # next-layer prediction: -1 off, 0 statistics only, k > 0 read ahead the k best predicted experts
                                   # (2026-10-05: 8 with 2 workers 70.2 -> 67.4 s on 3 questions, same answers; 12/16 read too much)
    [int]$PredictWorkers = 2,      # read-ahead jobs use at most this many of the I/O workers
    [int]$OffloadMinBatch = 1280,  # prompt batches of this many tokens go to the GPU (all experts of a layer copied over), smaller
                                   # ones to the CPU through the expert cache (GGML_OP_OFFLOAD_MIN_BATCH)
    [switch]$SoftWs,               # soft cap instead: the working set grows until Available RAM is ~80 MB (the watchdog kills it)
    [switch]$Think,                # thinking on (the template's default is xhigh); off by default
    [switch]$Lan,                  # listen on 0.0.0.0:8090 with the API key in api-key.txt (LAN + Tailscale; firewall: fw-qwen.ps1)
    [int]$MaxTokens = 2048,        # default reply cap when a request sets none (~10 min at 3 tok/s: a stop for runaway loops)
    [switch]$Stop
)
# the GGUF template has no reasoning_effort 'none': thinking is turned off with --reasoning off (enable_thinking=false)
$ErrorActionPreference = 'Stop'
$root  = 'C:\llama-qwen'
if ($Lan) {
    $ListenHost = '0.0.0.0'; $Port = 8090
    if (-not (Test-Path (Join-Path $root 'api-key.txt'))) { throw "missing $root\api-key.txt" }
}
$exe   = $Exe
# own build: no whole-file PrefetchVirtualMemory at load (the release llama.dll is byte-patched for the same effect)
$env:LLAMA_NO_MMAP_PREFETCH = '1'
if (-not $NoPrefetch) {
    $env:LLAMA_EXPERT_PREFETCH = '1'
    $env:LLAMA_EXPERT_PREFETCH_WORKERS = "$Workers"
    $env:LLAMA_EXPERT_PREFETCH_EVICT = "$Evict"
    $env:LLAMA_EXPERT_PREFETCH_TOUCH = "$Touch"
    $env:LLAMA_EXPERT_PREFETCH_STATS = $(if ($Stats) { '1' } else { '0' })
    $env:LLAMA_EXPERT_PREFETCH_TIMING = $(if ($Timing) { '1' } else { '0' })
    $env:LLAMA_EXPERT_PREFETCH_DIAG = $(if ($Diag) { '1' } else { '0' })
    $env:LLAMA_EXPERT_PREFETCH_TRACE = $Trace
    $env:LLAMA_GPU_KEEPALIVE = "$KeepAlive"
    $env:LLAMA_EXPERT_CACHE_GB = "$CacheGB"
    $env:LLAMA_EXPERT_CACHE_FILE = $CacheFile
    $env:LLAMA_EXPERT_CACHE_MMAP_GB = "$CacheMmapGB"
    $env:LLAMA_EXPERT_PREDICT = $(if ($Predict -ge 0) { '1' } else { '0' })
    $env:LLAMA_EXPERT_PREDICT_K = "$([Math]::Max(0, $Predict))"
    $env:LLAMA_EXPERT_PREDICT_WORKERS = "$PredictWorkers"
    $env:GGML_OP_OFFLOAD_MIN_BATCH = "$OffloadMinBatch"
    $env:LLAMA_EXPERT_CACHE_RELEASE_GB = "$PrefillReleaseGB"
}
$model = 'C:\models\Qwen3.8-Flash-Next\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null

function Stop-Qwen {
    Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root*" } | Stop-Process -Force
}
if ($Stop) { Stop-Qwen; 'stopped'; return }
Stop-Qwen

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$slog = Join-Path $logDir "server-$stamp.log"
$wlog = Join-Path $logDir "watchdog-$stamp.csv"
# --no-repack : repack copies CPU-side weights into anonymous RAM -> OOM with 16 GB
# -ot         : routed experts and the n-gram (per-layer) embedding stay mmapped on the CPU side, the rest goes to the GPU
# llama.dll is patched (PrefetchVirtualMemory -> XrefetchVirtualMemory) so loading does not prefetch whole files
$flags = @(
    '-m', $model, '--load-mode', 'mmap', '--no-repack',
    '-ngl', '99', '-ot', $Ot, '--fit', 'off', '-fa', 'on',
    '-c', $Ctx, '-ub', $Ubatch, '-np', '1', '-cram', '0',
    '-t', $Threads, '-tb', $Threads,
    '--jinja', '--no-warmup',
    '--host', $ListenHost, '--port', $Port, '--metrics'
)
$flags += @('--reasoning', $(if ($Think) { 'on' } else { 'off' }))
# sampling defaults (a request can override them): Unsloth's settings for this model
if ($Think) { $flags += @('--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0') }
else        { $flags += @('--temp', '0.7', '--top-p', '0.8', '--top-k', '20', '--min-p', '0', '--presence-penalty', '1.5') }
$flags += @('-n', $MaxTokens)
if ($Lan) { $flags += @('--api-key-file', (Join-Path $root 'api-key.txt')) }
if ($Extra) { $flags += ($Extra -split ' ') }
$p = Start-Process -FilePath $exe -ArgumentList $flags -RedirectStandardError $slog -RedirectStandardOutput "$slog.out" `
    -PassThru -WindowStyle Hidden
"pid $($p.Id)  log $slog  watchdog $wlog"
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class WsCap {
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
    // QUOTA_LIMITS_HARDWS_MIN_DISABLE (0x2) | QUOTA_LIMITS_HARDWS_MAX_ENABLE (0x4); soft: QUOTA_LIMITS_HARDWS_MAX_DISABLE (0x8)
    public static string Apply(int pid, long maxBytes, bool soft) {
        IntPtr h = OpenProcess(0x0100 | 0x0400, false, pid);   // PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION
        if (h == IntPtr.Zero) return "OpenProcess failed " + Marshal.GetLastWin32Error();
        bool ok = SetProcessWorkingSetSizeEx(h, (UIntPtr)(ulong)(64L << 20), (UIntPtr)(ulong)maxBytes, soft ? 0x2u | 0x8u : 0x2u | 0x4u);
        int err = Marshal.GetLastWin32Error();
        CloseHandle(h);
        return ok ? (soft ? "soft" : "hard") + " working-set cap set" : "SetProcessWorkingSetSizeEx failed " + err;
    }
}
'@
if ($MaxWsGB -gt 0) { "WS cap {0} GB: {1}" -f $MaxWsGB, [WsCap]::Apply($p.Id, [long]($MaxWsGB * 1GB), [bool]$SoftWs) }

function New-Pc($cat, $name, $inst = '') { New-Object System.Diagnostics.PerformanceCounter($cat, $name, $inst, $true) }
$pcAvail = New-Pc 'Memory' 'Available MBytes'
$pcCommitted = New-Pc 'Memory' 'Committed Bytes'
$pcLimit = New-Pc 'Memory' 'Commit Limit'
$pcRead = New-Pc 'PhysicalDisk' 'Disk Read Bytes/sec' '_Total'
$null = $pcRead.NextValue()
'time,availMB,commitFreeMB,privMB,wsMB,diskReadMBs,note' | Out-File $wlog -Encoding ascii
$t0 = Get-Date; $low = 0
while (-not $p.HasExited) {
    $avail = [int]$pcAvail.RawValue
    $cfree = [int](($pcLimit.RawValue - $pcCommitted.RawValue) / 1MB)
    $p.Refresh(); $priv = [int]($p.PrivateMemorySize64 / 1MB); $ws = [int]($p.WorkingSet64 / 1MB)
    if ($avail -lt $MinAvailMB) { $low++ } else { $low = 0 }
    $note = ''
    if ($cfree -lt $MinCommitFreeMB -or $low -ge 10) {
        $note = "KILL avail=$avail commitFree=$cfree priv=$priv"
        Stop-Process -Id $p.Id -Force
    }
    '{0:F0},{1},{2},{3},{4},{5:F0},{6}' -f ((Get-Date) - $t0).TotalSeconds, $avail, $cfree, $priv, $ws, ($pcRead.NextValue() / 1MB), $note |
        Out-File $wlog -Append -Encoding ascii
    if ($note) { Write-Warning $note; break }
    Start-Sleep -Milliseconds 1000
}
"server exited (code $($p.ExitCode)). tail of log:"
Get-Content $slog -Tail 15
