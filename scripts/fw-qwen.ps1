# One-time, run as administrator: let the local network (and a Tailscale-style VPN, 100.64.0.0/10) reach the Qwen
# server on port 8090. ASCII only.
# Created before the server first listens on 0.0.0.0, so Windows never shows its "allow access?" prompt (an unanswered
# prompt makes Windows add BLOCK rules for the exe, and a block rule wins over any allow rule).
param(
    [string]$Exe = 'C:\llama-qwen\build\bin\llama-server.exe',
    [string[]]$RemoteAddress = @('LocalSubnet', '100.64.0.0/10'),
    [string]$Log = 'C:\llama-qwen\logs\fw-qwen.log'
)
Start-Transcript -Path $Log -Force | Out-Null

Get-NetFirewallApplicationFilter -Program $Exe -ErrorAction SilentlyContinue | Get-NetFirewallRule |
    Where-Object { $_.Action -eq 'Block' } | ForEach-Object {
        "removing block rule: $($_.DisplayName) [$($_.Name)]"
        Remove-NetFirewallRule -Name $_.Name
    }

$name = 'Qwen 8090 (LAN + VPN)'
if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8090 `
        -RemoteAddress $RemoteAddress -Profile Private -Program $Exe | Out-Null
    "added allow rule: $name"
}

'rules for this exe now:'
Get-NetFirewallApplicationFilter -Program $Exe | Get-NetFirewallRule |
    Select-Object DisplayName, Action, Enabled, Profile | Format-Table -AutoSize | Out-String
Stop-Transcript | Out-Null
