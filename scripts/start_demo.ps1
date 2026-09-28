[CmdletBinding()]
param(
    [string]$EnvFile,
    [switch]$ValidateOnly,
    [switch]$NonInteractive,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $EnvFile) {
    $EnvFile = Join-Path $ProjectRoot ".env"
}

$AllowedKeys = @(
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "HF_HUB_OFFLINE",
    "PPMI_API_BASE_URL",
    "PPMI_LITERATURE_INDEX",
    "PPMI_MODEL_ARTIFACT",
    "PYTHON_EXE"
)

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split(@("="), 2, [System.StringSplitOptions]::None)
        $name = $parts[0].Trim()
        if ($AllowedKeys -notcontains $name) {
            continue
        }
        $values[$name] = $parts[1].Trim()
    }
    return $values
}

function Set-DotEnvValue {
    param([string]$Path, [string]$Name, [string]$Value)

    if ($Value.Contains("`r") -or $Value.Contains("`n")) {
        throw "$Name cannot contain a newline."
    }
    $lines = @()
    if (Test-Path -LiteralPath $Path) {
        $lines = @(Get-Content -LiteralPath $Path -Encoding UTF8)
    }
    $replacement = "$Name=$Value"
    $updated = $false
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match "^\s*$([regex]::Escape($Name))\s*=") {
            $lines[$index] = $replacement
            $updated = $true
        }
    }
    if (-not $updated) {
        $lines += $replacement
    }
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

function Read-SecretText {
    param([string]$Prompt)

    $secure = Read-Host $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Test-DeepSeekApiKeyFormat {
    param([string]$Value)
    return [bool]($Value -match '^sk-[A-Za-z0-9_-]{20,}$')
}

function Resolve-PythonExecutable {
    param([hashtable]$Configuration)

    $candidates = @()
    if ($Configuration["PYTHON_EXE"]) {
        $candidates += $Configuration["PYTHON_EXE"]
    }
    if ($env:CONDA_PREFIX) {
        $candidates += (Join-Path $env:CONDA_PREFIX "python.exe")
    }
    $candidates += (Join-Path $HOME "anaconda3\envs\pd-mci\python.exe")

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }
    throw "Python was not found. Set PYTHON_EXE in .env."
}

function Test-ListeningPort {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-HttpReady {
    param([string]$Url, [int]$Attempts = 30)
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Save-ProcessRecord {
    param([System.Diagnostics.Process]$Process, [string]$Kind, [string]$Path)
    @{
        pid = $Process.Id
        kind = $Kind
        executable = $Process.Path
        started_at = $Process.StartTime.ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Normalize-ProcessPathVariable {
    # Windows PowerShell 5.1 can receive both PATH and Path from parent tools.
    # Start-Process treats them as duplicate case-insensitive dictionary keys.
    $pathValue = $env:PATH
    Remove-Item Env:PATH -ErrorAction SilentlyContinue
    $env:Path = $pathValue
}

try {
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        $examplePath = Join-Path $ProjectRoot ".env.example"
        if (-not (Test-Path -LiteralPath $examplePath)) {
            throw ".env.example was not found."
        }
        Copy-Item -LiteralPath $examplePath -Destination $EnvFile
        Write-Host "Created local .env from .env.example."
    }

    $configuration = Read-DotEnv -Path $EnvFile
    if (-not (Test-DeepSeekApiKeyFormat -Value $configuration["DEEPSEEK_API_KEY"])) {
        if ($NonInteractive) {
            throw "DEEPSEEK_API_KEY must be a valid sk- key in .env."
        }
        $apiKey = Read-SecretText -Prompt "Enter your DeepSeek API key"
        if (-not (Test-DeepSeekApiKeyFormat -Value $apiKey)) {
            throw "The DeepSeek API key must be a valid sk- key."
        }
        Set-DotEnvValue -Path $EnvFile -Name "DEEPSEEK_API_KEY" -Value $apiKey
        $configuration["DEEPSEEK_API_KEY"] = $apiKey
        Write-Host "DeepSeek API key saved locally in .env (ignored by Git)."
    }

    $pythonExe = Resolve-PythonExecutable -Configuration $configuration
    if ($ValidateOnly) {
        Write-Host "Configuration valid. Python environment found; secret was not displayed."
        exit 0
    }

    if (Test-ListeningPort -Port 8000) {
        throw "Port 8000 is already in use. Run stop_demo.bat, then try again."
    }
    if (Test-ListeningPort -Port 8501) {
        throw "Port 8501 is already in use. Run stop_demo.bat, then try again."
    }

    foreach ($name in $AllowedKeys) {
        if ($configuration[$name] -and $name -ne "PYTHON_EXE") {
            [Environment]::SetEnvironmentVariable($name, $configuration[$name], "Process")
        }
    }
    if (-not $env:HF_HUB_OFFLINE) {
        $env:HF_HUB_OFFLINE = "1"
    }
    if (-not $env:PPMI_API_BASE_URL) {
        $env:PPMI_API_BASE_URL = "http://127.0.0.1:8000"
    }
    $env:PYTHONUNBUFFERED = "1"
    Normalize-ProcessPathVariable

    $runtimeDir = Join-Path $ProjectRoot ".runtime"
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

    $apiProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    Save-ProcessRecord -Process $apiProcess -Kind "api" -Path (Join-Path $runtimeDir "api.pid.json")

    if (-not (Wait-HttpReady -Url "http://127.0.0.1:8000/health")) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        throw "FastAPI did not become ready. Start it manually in PowerShell to view details."
    }

    $uiProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @("-m", "streamlit", "run", "ui\dashboard.py", "--server.headless", "true", "--server.port", "8501") `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    Save-ProcessRecord -Process $uiProcess -Kind "ui" -Path (Join-Path $runtimeDir "ui.pid.json")

    if (-not (Wait-HttpReady -Url "http://127.0.0.1:8501")) {
        Stop-Process -Id $uiProcess.Id -Force -ErrorAction SilentlyContinue
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        throw "Streamlit did not become ready. Start it manually in PowerShell to view details."
    }

    Write-Host "Demo started: http://127.0.0.1:8501"
    Write-Host "Use stop_demo.bat to stop both services."
    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:8501"
    }
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
