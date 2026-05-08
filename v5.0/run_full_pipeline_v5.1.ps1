# V7P3R AI v5.1 - Full Recalculation and Retraining Pipeline
# Automates the complete workflow from feature recalculation to trained model

Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "V7P3R AI v5.1 - Full Pipeline: Feature Expansion & Retraining" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan

$ErrorActionPreference = "Stop"
$startTime = Get-Date

# ============================================================================
# PHASE 1: Feature Recalculation (4-6 hours)
# ============================================================================
Write-Host "`nPHASE 1: Recalculating features on full dataset..." -ForegroundColor Yellow
Write-Host "Input:  data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" -ForegroundColor Gray
Write-Host "Output: data/final/v7p3r_ai_v5.1_expanded_features.jsonl" -ForegroundColor Gray
Write-Host "Features: 26 -> 92+ (tactical, rook placement, multi-move context)" -ForegroundColor Gray
Write-Host "Estimated time: 4-6 hours (230,930 positions)" -ForegroundColor Gray

$confirmRecalc = Read-Host "`nProceed with feature recalculation? (y/n)"
if ($confirmRecalc -ne 'y') {
    Write-Host "Aborted by user" -ForegroundColor Red
    exit 1
}

Write-Host "`nStarting feature recalculation..." -ForegroundColor Green
$recalcStart = Get-Date

python scripts/calculate_features.py `
    --input data/final/v7p3r_ai_v5_training_dataset_complete.jsonl `
    --output data/final/v7p3r_ai_v5.1_expanded_features.jsonl `
    --feature-set standard

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Feature recalculation failed" -ForegroundColor Red
    exit 1
}

$recalcDuration = (Get-Date) - $recalcStart
Write-Host "`nFeature recalculation complete in $($recalcDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green

# ============================================================================
# PHASE 2: Generate Splits (stratified 80/10/10)
# ============================================================================
Write-Host "`nPHASE 2: Generating train/val/test splits..." -ForegroundColor Yellow
Write-Host "Input:  data/final/v7p3r_ai_v5.1_expanded_features.jsonl" -ForegroundColor Gray
Write-Host "Output: data/analysis/splits_v5.1/" -ForegroundColor Gray
Write-Host "Split: 80% train, 10% val, 10% test (stratified by grade)" -ForegroundColor Gray

python scripts/split_dataset.py `
    --input data/final/v7p3r_ai_v5.1_expanded_features.jsonl `
    --output-dir data/analysis/splits_v5.1 `
    --train-ratio 0.8 `
    --val-ratio 0.1 `
    --test-ratio 0.1

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Dataset splitting failed" -ForegroundColor Red
    exit 1
}

Write-Host "Splits generated successfully" -ForegroundColor Green

# ============================================================================
# PHASE 3: Preprocessing (StandardScaler, OneHotEncoder)
# ============================================================================
Write-Host "`nPHASE 3: Preprocessing features..." -ForegroundColor Yellow
Write-Host "Input:  data/analysis/splits_v5.1/" -ForegroundColor Gray
Write-Host "Output: data/preprocessed_v5.1/" -ForegroundColor Gray
Write-Host "Transformations: StandardScaler (numerical), OneHotEncoder (categorical)" -ForegroundColor Gray

# Temporarily override split directory in preprocessing script
$env:V7P3R_SPLIT_DIR = "data/analysis/splits_v5.1"

python scripts/preprocess_dataset_v5.1.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Preprocessing failed" -ForegroundColor Red
    exit 1
}

Remove-Item Env:\V7P3R_SPLIT_DIR
Write-Host "Preprocessing complete" -ForegroundColor Green

# ============================================================================
# PHASE 4: Model Architecture Update
# ============================================================================
Write-Host "`nPHASE 4: Checking model architecture..." -ForegroundColor Yellow

# Get actual feature count from preprocessed data
$preprocessStats = Get-Content "data/preprocessed_v5.1/preprocessing_stats.json" | ConvertFrom-Json
$featureCount = $preprocessStats.total_features

Write-Host "Detected $featureCount total features (after one-hot encoding)" -ForegroundColor Gray
Write-Host "Model input_dim must be updated from 26 to $featureCount" -ForegroundColor Gray

$confirmModelUpdate = Read-Host "`nUpdate model architecture? (y/n)"
if ($confirmModelUpdate -ne 'y') {
    Write-Host "WARNING: Model architecture not updated - training will fail" -ForegroundColor Yellow
    Write-Host "Manually update src/model.py input_dim to $featureCount" -ForegroundColor Yellow
} else {
    Write-Host "Updating model.py..." -ForegroundColor Green
    
    # Read model file
    $modelPath = "src/model.py"
    $modelContent = Get-Content $modelPath -Raw
    
    # Update input_dim in config
    $modelContent = $modelContent -replace "input_dim: int = 26", "input_dim: int = $featureCount"
    
    # Save updated model
    Set-Content $modelPath $modelContent -Encoding UTF8
    
    Write-Host "Model architecture updated: input_dim = $featureCount" -ForegroundColor Green
}

# ============================================================================
# PHASE 5: Training (100 epochs with class weights)
# ============================================================================
Write-Host "`nPHASE 5: Training V7P3R AI v5.1..." -ForegroundColor Yellow
Write-Host "Input:  data/preprocessed_v5.1/" -ForegroundColor Gray
Write-Host "Output: checkpoints/, training_history.json" -ForegroundColor Gray
Write-Host "Config: 100 epochs, batch_size=256, class weights, early stopping" -ForegroundColor Gray

$confirmTrain = Read-Host "`nStart training? (y/n)"
if ($confirmTrain -ne 'y') {
    Write-Host "Training skipped" -ForegroundColor Yellow
    Write-Host "To train manually: python src/train.py --epochs 100 --batch-size 256 --data-dir data/preprocessed_v5.1" -ForegroundColor Gray
} else {
    Write-Host "`nStarting training..." -ForegroundColor Green
    $trainStart = Get-Date
    
    python src/train.py `
        --epochs 100 `
        --batch-size 256 `
        --data-dir data/preprocessed_v5.1 `
        --checkpoint-dir checkpoints_v5.1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nERROR: Training failed" -ForegroundColor Red
        exit 1
    }
    
    $trainDuration = (Get-Date) - $trainStart
    Write-Host "`nTraining complete in $($trainDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

# ============================================================================
# PHASE 6: Evaluation on Test Set
# ============================================================================
Write-Host "`nPHASE 6: Evaluating on test set..." -ForegroundColor Yellow

$confirmEval = Read-Host "`nRun evaluation? (y/n)"
if ($confirmEval -ne 'y') {
    Write-Host "Evaluation skipped" -ForegroundColor Yellow
} else {
    python src/evaluate.py `
        --checkpoint checkpoints_v5.1/best_model.pth `
        --data-dir data/preprocessed_v5.1 `
        --output-dir evaluation_results_v5.1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Evaluation complete - see evaluation_results_v5.1/evaluation_report.md" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Evaluation failed" -ForegroundColor Yellow
    }
}

# ============================================================================
# Summary
# ============================================================================
$totalDuration = (Get-Date) - $startTime

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "V7P3R AI v5.1 PIPELINE COMPLETE" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan

Write-Host "`nTotal time: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "`nResults:" -ForegroundColor White
Write-Host "  - Features expanded: 26 -> $featureCount" -ForegroundColor Gray
Write-Host "  - Dataset: 230,930 positions" -ForegroundColor Gray
Write-Host "  - Training checkpoint: checkpoints_v5.1/best_model.pth" -ForegroundColor Gray
Write-Host "  - Evaluation report: evaluation_results_v5.1/evaluation_report.md" -ForegroundColor Gray

Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  1. Review evaluation report for improved grade distribution" -ForegroundColor Gray
Write-Host "  2. Compare v5.1 vs v5.0 test accuracy (expect >54% vs 49%)" -ForegroundColor Gray
Write-Host "  3. Check confusion matrix for non-binary classification" -ForegroundColor Gray
Write-Host "  4. Update metrics dashboard with v5.1 session data" -ForegroundColor Gray
Write-Host ""
