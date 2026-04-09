param(
    [string]$DataDir = "data",
    [int]$Epochs = 25,
    [int]$BatchSize = 32,
    [int]$ImageSize = 112,
    [int]$Workers = 0,
  [string]$Backbone = "mobilenet_v3_small",
  [switch]$FastCPU
)

$ErrorActionPreference = "Stop"
$logsDir = Join-Path $PSScriptRoot "..\results\logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "Using Python: $pythonExe"

$classificationLog = Join-Path $logsDir "classification_overnight.log"
$metricLog = Join-Path $logsDir "metric_overnight.log"

$maxEvalPairs = 500
$evalEvery = 1

if ($FastCPU) {
  Write-Host "FastCPU mode enabled: reducing input size, batch size, and validation overhead."
  if ($ImageSize -eq 112) { $ImageSize = 96 }
  if ($BatchSize -eq 32) { $BatchSize = 24 }
  if ($Epochs -eq 25) { $Epochs = 18 }
  $maxEvalPairs = 200
  $evalEvery = 5
}

Write-Host "Starting classification training..."
& $pythonExe -u -m src.training.train_classification_local `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --image-size $ImageSize `
  --num-workers $Workers `
  --backbone $Backbone `
  --max-eval-pairs $maxEvalPairs `
  --eval-every $evalEvery `
  --eval-pairs `
  2>&1 | Tee-Object -FilePath $classificationLog

Write-Host "Starting metric-learning training..."
& $pythonExe -u -m src.training.train_metric_local `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --image-size $ImageSize `
  --num-workers $Workers `
  --backbone $Backbone `
  --max-eval-pairs $maxEvalPairs `
  --eval-every $evalEvery `
  --eval-pairs `
  2>&1 | Tee-Object -FilePath $metricLog

Write-Host "Overnight training complete."
Write-Host "Classification log: $classificationLog"
Write-Host "Metric log: $metricLog"
