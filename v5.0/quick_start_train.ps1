# V7P3R AI v5.0 - Quick Start Training Script
# Tests the training pipeline with a short run (2 epochs)

Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "V7P3R AI v5.0 - Quick Start Training Test" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host ("=" * 79) -ForegroundColor Cyan

Write-Host "`nThis script will:" -ForegroundColor White
Write-Host "  1. Verify preprocessed data exists" -ForegroundColor Gray
Write-Host "  2. Test model instantiation" -ForegroundColor Gray
Write-Host "  3. Run 2-epoch training test" -ForegroundColor Gray
Write-Host "  4. Validate training loop" -ForegroundColor Gray

# Change to project directory
$projectDir = "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"
Set-Location $projectDir

# Step 1: Check preprocessed data
Write-Host "`n[1/4] Checking preprocessed data..." -ForegroundColor Cyan

$dataFiles = @(
    "data\preprocessed\X_train.npy",
    "data\preprocessed\y_train_policy.npy",
    "data\preprocessed\y_train_value.npy",
    "data\preprocessed\X_val.npy",
    "data\preprocessed\y_val_policy.npy",
    "data\preprocessed\y_val_value.npy"
)

$allExist = $true
foreach ($file in $dataFiles) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1MB
        Write-Host "  ✅ $file ($($size.ToString('F2')) MB)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Missing: $file" -ForegroundColor Red
        $allExist = $false
    }
}

if (-not $allExist) {
    Write-Host "`n❌ Preprocessed data missing! Run:" -ForegroundColor Red
    Write-Host "  python scripts/preprocess_dataset.py" -ForegroundColor Yellow
    exit 1
}

# Step 2: Test model instantiation
Write-Host "`n[2/4] Testing model instantiation..." -ForegroundColor Cyan

python -c "import sys; sys.path.append('src'); from model import V7P3R_AI_v5; m = V7P3R_AI_v5(); print('✅ Model created successfully')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Model test failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Create test config (2 epochs)
Write-Host "`n[3/4] Creating test configuration..." -ForegroundColor Cyan

$testConfig = @"
# V7P3R AI v5.0 - Quick Test Configuration (2 epochs)

model:
  input_dim: 26
  shared_dims: [256, 256, 128, 64]
  policy_hidden: 64
  value_hidden: 32
  dropout: 0.3
  use_residuals: true

training:
  batch_size: 256
  num_workers: 4
  pin_memory: true
  
  epochs: 2                    # Quick test: 2 epochs only
  early_stopping_patience: 15
  save_every: 1
  
  policy_weight: 1.0
  value_weight: 0.1
  
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  
  lr_patience: 5
  lr_factor: 0.5
  min_lr: 0.000001
  
  huber_delta: 0.5
  
  checkpoint_dir: 'checkpoints_test'

evaluation:
  metrics:
    - policy_accuracy
    - policy_top2_accuracy
    - value_mae
    - value_correlation

logging:
  log_dir: 'logs_test'
  tensorboard: false
  save_predictions: false
"@

$testConfigPath = "configs/test_config.yaml"
Set-Content -Path $testConfigPath -Value $testConfig
Write-Host "  ✅ Test config created: $testConfigPath" -ForegroundColor Green

# Step 4: Run test training
Write-Host "`n[4/4] Running 2-epoch training test..." -ForegroundColor Cyan
Write-Host "  (This will take a few minutes...)" -ForegroundColor Gray

python src/train.py --config $testConfigPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Green; Write-Host ("=" * 79) -ForegroundColor Green
    Write-Host "✅ QUICK START TEST PASSED!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Green; Write-Host ("=" * 79) -ForegroundColor Green
    
    Write-Host "`nTest results:" -ForegroundColor White
    
    # Show checkpoint info
    if (Test-Path "checkpoints_test\latest_checkpoint.pth") {
        $checkpointSize = (Get-Item "checkpoints_test\latest_checkpoint.pth").Length / 1MB
        $sizeStr = $checkpointSize.ToString('F2')
        $message = "  📦 Checkpoint created: checkpoints_test\latest_checkpoint.pth (" + $sizeStr + " MB)"
        Write-Host $message -ForegroundColor Cyan
    }
    
    # Show training history
    if (Test-Path "checkpoints_test\training_history.json") {
        $history = Get-Content "checkpoints_test\training_history.json" | ConvertFrom-Json
        $finalMetrics = $history.metrics
        
        Write-Host "`n  📊 Final metrics (Epoch 2):" -ForegroundColor Cyan
        $trainLoss = $finalMetrics.train_loss[-1]
        $valLoss = $finalMetrics.val_loss[-1]
        $policyAcc = $finalMetrics.val_policy_acc[-1]
        
        $trainLossStr = $trainLoss.ToString('F4')
        $valLossStr = $valLoss.ToString('F4')
        $policyAccPct = [Math]::Round($policyAcc * 100, 2)
        
        Write-Host ("    Train Loss: " + $trainLossStr) -ForegroundColor Gray
        Write-Host ("    Val Loss: " + $valLossStr) -ForegroundColor Gray
        Write-Host ("    Policy Accuracy: " + $policyAccPct + "%") -ForegroundColor Gray
    }
    
    Write-Host "`n🚀 Ready for full training!" -ForegroundColor Yellow
    Write-Host "`nTo start full training (100 epochs):" -ForegroundColor White
    Write-Host "  python src/train.py --config configs/training_config.yaml" -ForegroundColor Cyan
    
} else {
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Red; Write-Host ("=" * 79) -ForegroundColor Red
    Write-Host "❌ TRAINING TEST FAILED" -ForegroundColor Red
    Write-Host "=" -NoNewline -ForegroundColor Red; Write-Host ("=" * 79) -ForegroundColor Red
    Write-Host "`nCheck the error messages above for details." -ForegroundColor Yellow
    exit 1
}
