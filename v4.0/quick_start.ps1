# V7P3RAI v4.0 Quick Start Script
# Sets up the environment and runs initial checks

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "V7P3RAI v4.0 - Quick Start Setup" -ForegroundColor Cyan
Write-Host "Multi-Agent Chess AI Enhancement Layer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to v4.0 directory
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK" -ForegroundColor Green
Write-Host ""

Write-Host "[2/6] Checking CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ PyTorch not installed yet (will install in next step)" -ForegroundColor Yellow
} else {
    Write-Host "✓ PyTorch OK" -ForegroundColor Green
}
Write-Host ""

Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

Write-Host "[4/6] Installing v7p3rai package..." -ForegroundColor Yellow
pip install -e . -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install package" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Package installed" -ForegroundColor Green
Write-Host ""

Write-Host "[5/6] Verifying project structure..." -ForegroundColor Yellow
$required_dirs = @(
    "src/agents",
    "src/core",
    "src/training",
    "config",
    "data/puzzles",
    "models/stage1_themes",
    "scripts",
    "docs"
)

$all_exist = $true
foreach ($dir in $required_dirs) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir MISSING" -ForegroundColor Red
        $all_exist = $false
    }
}

if (-not $all_exist) {
    Write-Host "ERROR: Some required directories are missing" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Project structure OK" -ForegroundColor Green
Write-Host ""

Write-Host "[6/6] Running sanity checks..." -ForegroundColor Yellow

# Test imports
python -c "from src.agents.v7p3r_themes_agent import V7P3RThemesAgent; print('✓ Themes agent import OK')" 2>&1
python -c "from src.core.agent_orchestrator import AgentOrchestrator; print('✓ Orchestrator import OK')" 2>&1
python -c "import chess; print('✓ python-chess import OK')" 2>&1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review the Master Plan:" -ForegroundColor White
Write-Host "   docs/V7P3RAI_V4.0_MASTER_PLAN.md" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Read Stage 1 Implementation Guide:" -ForegroundColor White
Write-Host "   docs/STAGE1_IMPLEMENTATION.md" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Prepare puzzle database:" -ForegroundColor White
Write-Host "   Link/copy 4M puzzles to: data/puzzles/4M_puzzle_library/" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Create puzzle dataset class:" -ForegroundColor White
Write-Host "   Implement: src/training/puzzle_dataset.py" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Port ChessState extractor from v3.0:" -ForegroundColor White
Write-Host "   Update: src/core/chess_state_extractor.py" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Start training:" -ForegroundColor White
Write-Host "   python scripts/stage1_train_themes.py --config config/training_config.json" -ForegroundColor Gray
Write-Host ""

Write-Host "For help: See README.md" -ForegroundColor Yellow
Write-Host ""
