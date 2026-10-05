# Send a few fixed Japanese prompts to the local server and record speed. ASCII only (prompts are \u escapes in JSON).
param([int]$Port = 8091, [int]$MaxTokens = 128, [string]$Tag = 'run', [double]$Temp = 0)
$out = "C:\llama-qwen\bench\$Tag-$(Get-Date -Format 'HHmmss').jsonl"
# 1) Fuji in two sentences  2) SSD vs HDD in three bullet points  3) a short haiku-like poem about autumn
$prompts = @(
    '\u5bcc\u58eb\u5c71\u306b\u3064\u3044\u3066\u0032\u6587\u3067\u6559\u3048\u3066\u3002',
    'SSD\u3068HDD\u306e\u9055\u3044\u3092\u7b87\u6761\u66f8\u304d\u0033\u3064\u3067\u8aac\u660e\u3057\u3066\u3002',
    '\u79cb\u306e\u5915\u66ae\u308c\u3092\u984c\u306b\u3057\u305f\u77ed\u3044\u8a69\u3092\u66f8\u3044\u3066\u3002'
)
foreach ($q in $prompts) {
    $body = '{"messages":[{"role":"user","content":"' + $q + '"}],"max_tokens":' + $MaxTokens + ',"temperature":' + $Temp + ',"seed":1}'
    $t = Get-Date
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 3600
    $tm = $r.timings
    $rec = [ordered]@{ wall_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1); prompt_n = $tm.prompt_n; prompt_tps = $tm.prompt_per_second;
        gen_n = $tm.predicted_n; gen_tps = $tm.predicted_per_second; text = $r.choices[0].message.content }
    ($rec | ConvertTo-Json -Compress) | Out-File $out -Append -Encoding utf8
    '{0,6}s  prompt {1,4} tok @ {2,6:N1} t/s  gen {3,4} tok @ {4,5:N2} t/s' -f $rec.wall_s, $rec.prompt_n, $rec.prompt_tps, $rec.gen_n, $rec.gen_tps
}
"saved $out"
