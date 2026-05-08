# Monitor Stockfish Grading Progress
# Run this script in a separate PowerShell window to track progress

$outputFile = "data/training/all_pgn_graded_depth15.jsonl"
$totalPositions = 210054

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Stockfish Grading Progress Monitor" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total Positions: $totalPositions" -ForegroundColor Yellow
Write-Host "Target: Depth 15, MultiPV 5" -ForegroundColor Yellow
Write-Host "Estimated Time: ~29 hours" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop monitoring (grading will continue)" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date

while ($true) {
    if (Test-Path $outputFile) {
        $lineCount = (Get-Content $outputFile | Measure-Object -Line).Lines
        $fileSize = (Get-Item $outputFile).Length / 1MB
        $percentComplete = ($lineCount / $totalPositions) * 100
        
        $elapsed = (Get-Date) - $startTime
        
        if ($lineCount -gt 0) {
            $posPerSec = $lineCount / $elapsed.TotalSeconds
            $remaining = ($totalPositions - $lineCount) / $posPerSec
            $remainingHours = [math]::Floor($remaining / 3600)
            $remainingMins = [math]::Floor(($remaining % 3600) / 60)
            $eta = "~$remainingHours hr $remainingMins min"
        } else {
            $posPerSec = 0
            $eta = "Calculating..."
        }
        
        Clear-Host
        Write-Host "============================================" -ForegroundColor Cyan
        Write-Host "  Stockfish Grading Progress Monitor" -ForegroundColor Cyan
        Write-Host "============================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Positions Graded: " -NoNewline
        Write-Host "$lineCount / $totalPositions" -ForegroundColor Green
        Write-Host "Progress: " -NoNewline
        Write-Host ("{0:N2}%" -f $percentComplete) -ForegroundColor Green
        Write-Host "File Size: " -NoNewline
        Write-Host ("{0:N2} MB" -f $fileSize) -ForegroundColor Green
        Write-Host ""
        Write-Host "Speed: " -NoNewline
        Write-Host ("{0:N2} pos/sec" -f $posPerSec) -ForegroundColor Yellow
        Write-Host "Elapsed: " -NoNewline
        Write-Host ("{0:hh}:{0:mm}:{0:ss}" -f $elapsed) -ForegroundColor Yellow
        Write-Host "ETA: " -NoNewline
        Write-Host $eta -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Last Update: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
    } else {
        Write-Host "Waiting for output file to be created..." -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds 5
}
