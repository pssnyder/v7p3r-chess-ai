# V7P3R AI v5.2 - Wide Architecture Training Script
# Breaks through the 45% plateau with 3x wider network

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "V7P3R AI v5.2 - WIDE ARCHITECTURE TRAINING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Architecture Upgrade:" -ForegroundColor Yellow
Write-Host "  v5.1: 325 -> 256 -> 256 -> 128 -> 64  (239k params)" -ForegroundColor White
Write-Host "  v5.2: 325 -> 512 -> 512 -> 256 -> 128 (953k params)" -ForegroundColor Green
Write-Host ""
Write-Host "  Parameter increase: 3x" -ForegroundColor Green
Write-Host "  Expected accuracy: 50-53% (from 45%)" -ForegroundColor Green
Write-Host "  Training time: ~25min (from 13min)" -ForegroundColor White
Write-Host ""

# Check if preprocessed data exists
if (-not (Test-Path "data\preprocessed_v5.1\X_train.npy")) {
    Write-Host "ERROR: Preprocessed data not found!" -ForegroundColor Red
    Write-Host "Run the full pipeline first: .\run_full_pipeline_v5.1_temporal.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  Config: configs\training_config_v5.2_wide.yaml" -ForegroundColor White
Write-Host "  Data: data\preprocessed_v5.1\ (323,656 positions)" -ForegroundColor White
Write-Host "  Output: models\v5.2_wide\" -ForegroundColor White
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Path "models\v5.2_wide" -Force | Out-Null

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Start timer
$startTime = Get-Date

# Run training
python src\train.py `
    --config configs\training_config_v5.2_wide.yaml `
    --data-dir data\preprocessed_v5.1

$exitCode = $LASTEXITCODE

# Calculate duration
$endTime = Get-Date
$duration = $endTime - $startTime
$minutes = [math]::Floor($duration.TotalMinutes)
$seconds = $duration.Seconds

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "Duration: $minutes min $seconds sec" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Evaluate: python src\evaluate.py --checkpoint models\v5.2_wide\best_model.pth --data-dir data\preprocessed_v5.1 --output-dir evaluation_results_v5.2_wide" -ForegroundColor White
    Write-Host "  2. Compare with v5.1 results in evaluation_results_v5.1_tpf\" -ForegroundColor White
    Write-Host "  3. Update MODEL_METRICS_GUIDE.html with new session" -ForegroundColor White
} else {
    Write-Host "Training failed with exit code: $exitCode" -ForegroundColor Red
}

Write-Host "============================================================" -ForegroundColor Cyan
