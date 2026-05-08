# V7P3R AI v5.0 - Full Training Run Script (100 epochs)
# This script runs the complete training pipeline and captures metrics snapshots

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                    V7P3R AI v5.0 - Full Training Run                          " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date

Write-Host "[" -NoNewline
Write-Host "INFO" -ForegroundColor Green -NoNewline
Write-Host "] Training Configuration:" -ForegroundColor White
Write-Host "  - Config:           training_config.yaml"
Write-Host "  - Epochs:           100"
Write-Host "  - Batch Size:       256"
Write-Host "  - Learning Rate:    0.001"
Write-Host "  - Dataset:          230,930 positions"
Write-Host "  - Est. Duration:    20-25 minutes (CPU)"
Write-Host ""

Write-Host "[" -NoNewline
Write-Host "START" -ForegroundColor Yellow -NoNewline
Write-Host "] Starting training at: " -NoNewline -ForegroundColor White
Write-Host $startTime.ToString("yyyy-MM-dd HH:mm:ss") -ForegroundColor Cyan
Write-Host ""

# Run training
Write-Host "================================================================================" -ForegroundColor White
python src/train.py --config configs/training_config.yaml

$trainExitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "================================================================================" -ForegroundColor White

if ($trainExitCode -eq 0) {
    Write-Host "[" -NoNewline
    Write-Host "SUCCESS" -ForegroundColor Green -NoNewline
    Write-Host "] Training completed successfully!" -ForegroundColor White
    
    $minutes = [math]::Floor($duration.TotalMinutes)
    $seconds = $duration.Seconds
    Write-Host "  - Total Duration:   $minutes minutes $seconds seconds"
    Write-Host "  - Completed at:     " -NoNewline
    Write-Host $endTime.ToString("yyyy-MM-dd HH:mm:ss") -ForegroundColor Cyan
    Write-Host ""
    
    # Snapshot metrics
    Write-Host "[" -NoNewline
    Write-Host "SNAPSHOT" -ForegroundColor Yellow -NoNewline
    Write-Host "] Capturing training metrics..." -ForegroundColor White
    Write-Host ""
    
    python scripts/snapshot_metrics.py --checkpoint checkpoints/best_model.pth --session-name "Full Training Session 1"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[" -NoNewline
        Write-Host "COMPLETE" -ForegroundColor Green -NoNewline
        Write-Host "] Metrics snapshot saved successfully!" -ForegroundColor White
        Write-Host ""
        
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "                              Next Steps                                       " -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "1. Review training metrics:"
        Write-Host "   " -NoNewline
        Write-Host "Start-Process 'docs\MODEL_METRICS_GUIDE.html'" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "2. Update dashboard with full training data:"
        Write-Host "   " -NoNewline
        Write-Host "python scripts/update_metrics_dashboard.py --checkpoint checkpoints/best_model.pth --session-name 'Full Training Session 1'" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "3. Evaluate on test set:"
        Write-Host "   " -NoNewline
        Write-Host "python src/evaluate.py --checkpoint checkpoints/best_model.pth" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "4. View historical trends:"
        Write-Host "   " -NoNewline
        Write-Host "cat metrics_snapshots/progression_report.txt" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "================================================================================" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "[" -NoNewline
        Write-Host "WARNING" -ForegroundColor Yellow -NoNewline
        Write-Host "] Metrics snapshot failed, but training completed successfully" -ForegroundColor White
    }
    
} else {
    Write-Host "[" -NoNewline
    Write-Host "FAILED" -ForegroundColor Red -NoNewline
    Write-Host "] Training failed with exit code: $trainExitCode" -ForegroundColor White
    Write-Host ""
    Write-Host "Check logs above for error details."
}

Write-Host ""
Write-Host "Script execution time: $($duration.TotalMinutes.ToString('0.0')) minutes"
Write-Host ""
