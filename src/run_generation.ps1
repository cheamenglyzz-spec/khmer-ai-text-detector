# Runs step 2 (src/generate_ai.py) until every assigned row is generated.
#
# Free-tier daily quotas stop a generator for the rest of a run (gpt-oss-120b allows
# ~200k tokens/day on Groq, about 150 texts). generate_ai.py resumes from
# data/generated/ai_raw.jsonl, so this loop simply reruns it every $PauseMinutes until it
# reports nothing left to do. All output goes to data/generated/full_run.log.
#
# Start (from the repo root):  powershell -ExecutionPolicy Bypass -File src\run_generation.ps1
# Stop: close this window or press Ctrl+C. Saved texts are kept; starting again resumes.
param([int]$PauseMinutes = 30)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $root "data\generated\full_run.log"
$host.UI.RawUI.WindowTitle = "Khmer AI text generation (step 2)"

$run = 0
while ($true) {
    $run++
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] run $run started - progress: Get-Content data\generated\full_run.log -Wait -Tail 5"
    # cmd passes Python's UTF-8 bytes through untouched; PowerShell 5.1 redirection would re-encode them
    cmd /c "echo. >> `"$log`" & echo === run $run started $stamp === >> `"$log`" & .venv\Scripts\python.exe -I src\generate_ai.py >> `"$log`" 2>&1"
    $code = $LASTEXITCODE
    $tail = Get-Content $log -Tail 40 -Encoding UTF8
    if ($tail -match "Nothing to do") {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] all rows generated, done."
        break
    }
    if ($code -eq 130) {
        Write-Host "Interrupted, stopping."
        break
    }
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] run $run ended (exit $code); next run in $PauseMinutes min"
    Start-Sleep -Seconds ($PauseMinutes * 60)
}
