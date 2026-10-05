# Long-prompt benchmark: summarize the first N characters of a Japanese text file. ASCII only.
param([int]$Port = 8091, [int]$Chars = 6000, [int]$MaxTokens = 100, [string]$Tag = 'long', [string]$Source = '')
# the text to summarize (the 2026-10-05 measurements used the first 6000 characters of a Japanese report)
if (-not $Source) { throw 'pass -Source <a UTF-8 text file>' }
$text = [IO.File]::ReadAllText($Source, [Text.Encoding]::UTF8)
if ($text.Length -gt $Chars) { $text = $text.Substring(0, $Chars) }
# "Summarize the following document in three lines." (as \u escapes to keep this file ASCII)
$ask = [regex]::Unescape('\u6b21\u306e\u6587\u66f8\u3092\u0033\u884c\u3067\u8981\u7d04\u3057\u3066\u304f\u3060\u3055\u3044\u3002')
$body = @{ messages = @(@{ role = 'user'; content = "$ask`n`n$text" }); max_tokens = $MaxTokens; temperature = 0; seed = 1 } | ConvertTo-Json -Depth 5
$t = Get-Date
$r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' `
    -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 7200
$tm = $r.timings
$rec = [ordered]@{ tag = $Tag; wall_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1); prompt_n = $tm.prompt_n; prompt_tps = $tm.prompt_per_second;
    gen_n = $tm.predicted_n; gen_tps = $tm.predicted_per_second; text = $r.choices[0].message.content }
($rec | ConvertTo-Json -Compress) | Out-File "C:\llama-qwen\bench\$Tag-$(Get-Date -Format 'HHmmss').jsonl" -Append -Encoding utf8
'{0,6}s  prompt {1,5} tok @ {2,6:N1} t/s  gen {3,4} tok @ {4,5:N2} t/s' -f $rec.wall_s, $rec.prompt_n, $rec.prompt_tps, $rec.gen_n, $rec.gen_tps
