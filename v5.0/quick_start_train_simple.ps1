# V7P3R AI v5.0 - Quick Start Training Test
# Simple version without emoji encoding issues

Write-Host "=" "================================================================" -NoNewline
Write-Host ""
Write-Host "V7P3R AI v5.0 - Quick Start Training Test"
Write-Host "================================================================="

Write-Host ""
Write-Host "This script will:"
Write-Host "  1. Verify preprocessed data exists"
Write-Host "  2. Test model instantiation"
Write-Host "  3. Run 2-epoch training test"
Write-Host "  4. Validate training loop"

# Change to project directory
$projectDir = "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"
Set-Location $projectDir

# Step 1: Check preprocessed data
Write-Host ""
Write-Host "[1/4] Checking preprocessed data..."

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
        $sizeStr = "{0:F2}" -f $size
        Write-Host "  [OK] $file ($sizeStr MB)"
    } else {
        Write-Host "  [ERROR] Missing: $file"
        $allExist = $false
    }
}

if (-not $allExist) {
    Write-Host ""
    Write-Host "[ERROR] Preprocessed data missing! Run:"
    Write-Host "  python scripts/preprocess_dataset.py"
    exit 1
}

# Step 2: Test model instantiation
Write-Host ""
Write-Host "[2/4] Testing model instantiation..."

python -c "import sys; sys.path.append('src'); from model import V7P3R_AI_v5; m = V7P3R_AI_v5(); print('[OK] Model created successfully')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Model test failed!"
    exit 1
}

# Step 3: Create test config (2 epochs)
Write-Host ""
Write-Host "[3/4] Creating test configuration..."

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
  
  epochs: 2
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
Write-Host "  [OK] Test config created: $testConfigPath"

# Step 4: Run test training
Write-Host ""
Write-Host "[4/4] Running 2-epoch training test..."
Write-Host "  (This will take a few minutes...)"
Write-Host ""

python src/train.py --config $testConfigPath

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================="
    Write-Host "QUICK START TEST PASSED!"
    Write-Host "================================================================="
    
    Write-Host ""
    Write-Host "Test results:"
    
    # Show checkpoint info
    if (Test-Path "checkpoints_test\latest_checkpoint.pth") {
        $checkpointSize = (Get-Item "checkpoints_test\latest_checkpoint.pth").Length / 1MB
        $sizeStr = "{0:F2}" -f $checkpointSize
        Write-Host "  Checkpoint created: checkpoints_test\latest_checkpoint.pth ($sizeStr MB)"
    }
    
    # Show training history
    if (Test-Path "checkpoints_test\training_history.json") {
        $history = Get-Content "checkpoints_test\training_history.json" | ConvertFrom-Json
        $finalMetrics = $history.metrics
        
        Write-Host ""
        Write-Host "  Final metrics (Epoch 2):"
        $trainLoss = $finalMetrics.train_loss[-1]
        $valLoss = $finalMetrics.val_loss[-1]
        $policyAcc = $finalMetrics.val_policy_acc[-1]
        
        $trainLossStr = "{0:F4}" -f $trainLoss
        $valLossStr = "{0:F4}" -f $valLoss
        $policyAccPct = "{0:F2}" -f ($policyAcc * 100)
        
        Write-Host "    Train Loss: $trainLossStr"
        Write-Host "    Val Loss: $valLossStr"
        Write-Host "    Policy Accuracy: $policyAccPct%"
    }
    
    Write-Host ""
    Write-Host "Ready for full training!"
    Write-Host ""
    Write-Host "To start full training (100 epochs):"
    Write-Host "  python src/train.py --config configs/training_config.yaml"
    
} else {
    Write-Host ""
    Write-Host "================================================================="
    Write-Host "TRAINING TEST FAILED"
    Write-Host "================================================================="
    Write-Host ""
    Write-Host "Check the error messages above for details."
    exit 1
}
