# Prompt-processing speed by prompt length: the first N characters of a UTF-8 text, one token generated, no prompt
# cache (every request processes its whole prompt). ASCII only. One line per size to -Out.
param([int]$Port = 8091, [int[]]$Chars = @(100, 200, 400, 800, 1600), [Parameter(Mandatory)][string]$Source,
      [string]$Tag = 'sweep', [string]$Out = 'C:\llama-qwen\bench\promptsweep.txt')
$text = [IO.File]::ReadAllText($Source, [Text.Encoding]::UTF8)
foreach ($n in $Chars) {
    $body = @{ messages = @(@{ role = 'user'; content = $text.Substring(0, [Math]::Min($n, $text.Length)) }); max_tokens = 1;
               temperature = 0; cache_prompt = $false } | ConvertTo-Json -Depth 5
    $t = Get-Date
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 3600
    '{0} chars {1,5}: prompt {2,5} tok in {3,6:N1} s = {4,6:N1} t/s (wall {5:N1} s)' -f $Tag, $n, $r.timings.prompt_n,
        ($r.timings.prompt_ms / 1000), $r.timings.prompt_per_second, ((Get-Date) - $t).TotalSeconds | Out-File $Out -Append -Encoding ascii
}
