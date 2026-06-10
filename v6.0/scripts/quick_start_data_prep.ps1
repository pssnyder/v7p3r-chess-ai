# V7P3R AI v6.0 - Quick Start: Data Preparation
# Runs Stage 1 data filtering and graph building

Write-Host "============================================================"
Write-Host "V7P3R AI v6.0 - Stage 1 Data Preparation"
Write-Host "============================================================"
Write-Host ""

# Step 1: Filter dataset
Write-Host "Step 1: Filtering dataset for binary classification..."
Write-Host "  - Input: v5.3 merged dataset (6.3M positions)"
Write-Host "  - Output: Good positions (G0 + filtered G1) + Bad positions (G2-G5)"
Write-Host ""

python scripts\stage1\filter_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Filtering failed!"
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host ""

# Step 2: Build transposition graph
Write-Host "Step 2: Building transposition graph..."
Write-Host "  - Input: Filtered good positions"
Write-Host "  - Output: Position similarity network"
Write-Host ""

python scripts\stage1\build_graph.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Graph building failed!"
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "✅ DATA PREPARATION COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review filtered dataset statistics"
Write-Host "  2. Implement Stage 1 training (graph NN)"
Write-Host "  3. Train policy network"
Write-Host ""
