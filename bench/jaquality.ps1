# Japanese quality check: varied prompts at the server's default sampling (no temperature in the request), then count
# characters that never appear in Japanese text (simplified-only Chinese, Hangul) per answer. ASCII only:
# the prompts live in jaquality-prompts.txt (UTF-8, one per line).
param([int]$Port = 8091, [int]$MaxTokens = 400, [string]$Tag = 'jaq', [int]$Seed = 1)
$prompts = Get-Content 'C:\llama-qwen\bench\jaquality-prompts.txt' -Encoding UTF8 | Where-Object { $_.Trim() }
$out = "C:\llama-qwen\bench\$Tag-$(Get-Date -Format 'HHmmss').jsonl"
# characters that Shift_JIS (CP932) cannot encode are not used in Japanese: simplified-only hanzi, Hangul, etc.
$sjis = [Text.Encoding]::GetEncoding(932, [Text.EncoderFallback]::ExceptionFallback, [Text.DecoderFallback]::ExceptionFallback)
function Get-Foreign([string]$s) {
    $bad = New-Object System.Collections.Generic.List[string]
    foreach ($ch in $s.ToCharArray()) {
        $c = [int]$ch
        if ($c -lt 0x2E80 -or [char]::IsSurrogate($ch)) { continue }   # ASCII, Latin, symbols: not the point here
        if ($c -eq 0x301C) { continue }   # wave dash: Japanese, but CP932 only knows the fullwidth tilde U+FF5E for it
        try { [void]$sjis.GetBytes([string]$ch) } catch { $bad.Add([string]$ch) }
    }
    return $bad
}
$n = 0
foreach ($q in $prompts) {
    $n++
    # [string]: lines from Get-Content carry PSPath etc. as note properties, which ConvertTo-Json would serialize
    $body = @{ messages = @(@{ role = 'user'; content = [string]$q }); max_tokens = $MaxTokens; seed = $Seed } | ConvertTo-Json -Depth 5
    $t = Get-Date
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 3600
    } catch { "{0,2}: request failed: {1}" -f $n, $_.Exception.Message; continue }
    $text = $r.choices[0].message.content
    $bad = Get-Foreign $text
    $rec = [ordered]@{ n = $n; wall_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1); gen_n = $r.timings.predicted_n;
        gen_tps = $r.timings.predicted_per_second; finish = $r.choices[0].finish_reason; foreign = ($bad -join ''); prompt = $q; text = $text }
    ($rec | ConvertTo-Json -Compress) | Out-File $out -Append -Encoding utf8
    '{0,2}: {1,6}s  gen {2,4} tok @ {3,5:N2} t/s  finish={4}  foreign={5}' -f $n, $rec.wall_s, $rec.gen_n, $rec.gen_tps, $rec.finish, $bad.Count
}
"saved $out"
