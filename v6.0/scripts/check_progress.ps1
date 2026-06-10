# Monitor Stage 1 data preparation progress

Write-Host "V7P3R AI v6.0 - Data Preparation Progress Monitor"
Write-Host "=================================================="
Write-Host ""

# Check if filtering is complete
$goodFile = "data\stage1\good_positions.jsonl"
$badFile = "data\stage1\bad_positions.jsonl"

if (Test-Path $goodFile) {
    $goodSize = (Get-Item $goodFile).Length / 1GB
    $goodLines = (Get-Content $goodFile | Measure-Object -Line).Lines
    Write-Host "✅ Good positions file: $goodLines positions ($($goodSize.ToString('F2')) GB)"
} else {
    Write-Host "⏳ Good positions file: Not created yet"
}

if (Test-Path $badFile) {
    $badSize = (Get-Item $badFile).Length / 1MB
    $badLines = (Get-Content $badFile | Measure-Object -Line).Lines
    Write-Host "✅ Bad positions file: $badLines positions ($($badSize.ToString('F2')) MB)"
} else {
    Write-Host "⏳ Bad positions file: Not created yet"
}

Write-Host ""

# Check if graph is complete
$graphFile = "data\stage1\transposition_graph.pkl"

if (Test-Path $graphFile) {
    $graphSize = (Get-Item $graphFile).Length / 1MB
    Write-Host "✅ Transposition graph: $($graphSize.ToString('F2')) MB"
} else {
    Write-Host "⏳ Transposition graph: Not created yet"
}

Write-Host ""
Write-Host "Expected completion:"
Write-Host "  - Good positions: ~5.7M records (~20 GB)"
Write-Host "  - Bad positions: ~33k records (~100 MB)"
Write-Host "  - Transposition graph: ~500 MB"
Write-Host ""
