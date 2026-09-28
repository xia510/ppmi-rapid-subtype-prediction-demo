[CmdletBinding()]
param([switch]$Quiet)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot ".runtime"

function Stop-RecordedProcess {
    param([string]$RecordPath)

    if (-not (Test-Path -LiteralPath $RecordPath)) {
        return $false
    }
    $record = Get-Content -LiteralPath $RecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $RecordPath -Force
        return $false
    }

    $recordedStart = [DateTime]::Parse([string]$record.started_at).ToUniversalTime()
    $actualStart = $process.StartTime.ToUniversalTime()
    if ([Math]::Abs(($actualStart - $recordedStart).TotalSeconds) -gt 2) {
        throw "Refusing to stop PID $($process.Id): its start time does not match the record."
    }
    if ($record.executable -and $process.Path -ne [string]$record.executable) {
        throw "Refusing to stop PID $($process.Id): its executable does not match the record."
    }

    Stop-Process -Id $process.Id -Force
    Remove-Item -LiteralPath $RecordPath -Force
    return $true
}

try {
    $stoppedUi = Stop-RecordedProcess `
        -RecordPath (Join-Path $RuntimeDir "ui.pid.json")
    $stoppedApi = Stop-RecordedProcess `
        -RecordPath (Join-Path $RuntimeDir "api.pid.json")

    if (-not $Quiet) {
        if ($stoppedUi -or $stoppedApi) {
            Write-Host "Demo services stopped."
        }
        else {
            Write-Host "No launcher-managed demo services were running."
        }
    }
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
