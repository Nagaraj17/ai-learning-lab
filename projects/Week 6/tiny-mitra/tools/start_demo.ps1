param([switch]$Restart)
$mitraRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$mitraPython = Join-Path $mitraRoot '.venv-repro/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $mitraPython)) { $mitraPython = Join-Path $mitraRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $mitraPython)) { throw 'Create the project virtual environment first; see README.md.' }
$mitraPattern = '*' + $mitraRoot + '\.venv\Scripts\python.exe*app.py --port 7860*'
$mitraReproPattern = '*' + $mitraRoot + '\.venv-repro\Scripts\python.exe*app.py --port 7860*'
$mitraWrappers = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like $mitraPattern -or $_.CommandLine -like $mitraReproPattern) })
if ($mitraWrappers.Count -gt 0 -and -not $Restart) {
    Write-Output 'Project demo is already running at http://127.0.0.1:7860. Use -Restart after source changes.'
    return
}
foreach ($mitraWrapper in $mitraWrappers) {
    $mitraChildren = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mitraWrapper.ProcessId -and $_.CommandLine -like '*app.py --port 7860*' }
    foreach ($mitraChild in $mitraChildren) { Stop-Process -Id $mitraChild.ProcessId -ErrorAction SilentlyContinue }
    Stop-Process -Id $mitraWrapper.ProcessId -ErrorAction SilentlyContinue
}
$mitraResults = Join-Path $mitraRoot 'artifacts/results'
New-Item -ItemType Directory -Path $mitraResults -Force | Out-Null
$mitraServer = Start-Process -FilePath $mitraPython -ArgumentList @('-u','app.py','--port','7860') -WorkingDirectory $mitraRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $mitraResults 'server.stdout.log') -RedirectStandardError (Join-Path $mitraResults 'server.stderr.log')
@{pid=$mitraServer.Id;command='app.py --port 7860';started=(Get-Date).ToString('o');app_source_sha256=(Get-FileHash -LiteralPath (Join-Path $mitraRoot 'app.py') -Algorithm SHA256).Hash} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $mitraResults 'server_process.json')
Write-Output "Started project demo process $($mitraServer.Id); checking readiness."
$mitraTimer = [System.Diagnostics.Stopwatch]::StartNew()
while ($mitraTimer.Elapsed.TotalSeconds -lt 45) {
    try {
        $mitraResponse = Invoke-WebRequest -Uri 'http://127.0.0.1:7860/config' -NoProxy -TimeoutSec 2
        if ($mitraResponse.StatusCode -eq 200) {
            Write-Output 'Demo is ready: http://127.0.0.1:7860'
            return
        }
    } catch { }
    if ($mitraServer.HasExited) { throw 'Demo exited during startup; inspect artifacts/results/server.stderr.log.' }
    Start-Sleep -Seconds 2
}
Write-Warning 'Process is running but readiness is not confirmed yet. Inspect the server logs before opening the demo.'
