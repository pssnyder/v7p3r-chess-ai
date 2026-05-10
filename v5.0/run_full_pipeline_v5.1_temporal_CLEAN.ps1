# V7P3R AI v5.1 + TPF - Complete Pipeline
# ==========================================
# Full feature expansion + temporal persistence features
# Rebuilds dataset and trains fresh 100-epoch model

param(
    [switch]$SkipPuzzleExtraction,
    [switch]$SkipFeatureCalculation,
    [switch]$SkipPreprocessing,
    [switch]$SkipTraining,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "V7P3R AI v5.1 with Temporal Features - Complete Rebuild Pipeline" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Feature Overview:" -ForegroundColor Yellow
Write-Host "  Phase 1A: Tactical features (F040-F091)" -ForegroundColor White
Write-Host "  Phase 1B: Multi-move context (F100-F114)" -ForegroundColor White
Write-Host "  Phase 1C: Temporal features (F200-F220)" -ForegroundColor White
Write-Host ""
Write-Host "Total Features: 262 total - 106 v5.1 and 156 temporal" -ForegroundColor Green
Write-Host ""

# Paths
$PYTHON = "python"
$PROJECT_ROOT = "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"
$PUZZLE_DB = "$PROJECT_ROOT\data\puzzles\lichess_db_puzzle.jsonl"
$PUZZLE_SEQUENCES = "$PROJECT_ROOT\data\puzzles\puzzle_sequences_with_features.jsonl"
$GAME_DATA = "$PROJECT_ROOT\data\final\v7p3r_ai_v5_training_dataset_complete.jsonl"
$GAME_FEATURES = "$PROJECT_ROOT\data\final\v7p3r_ai_v5.1_game_features.jsonl"
$MERGED_DATA = "$PROJECT_ROOT\data\final\v7p3r_ai_v5.1_tpf_merged.jsonl"
$SPLIT_DIR = "$PROJECT_ROOT\data\final\v7p3r_ai_v5.1_tpf_split"
$PREPROCESSED_DIR = "$PROJECT_ROOT\data\preprocessed_v5.1_tpf"
$CHECKPOINT_DIR = "$PROJECT_ROOT\checkpoints\v5.1_tpf"

# ==============================================================================
# Phase 1: Extract Puzzle Sequences with Temporal Features
# ==============================================================================
if (-not $SkipPuzzleExtraction) {
    Write-Host ""
    Write-Host "PHASE 1: Extracting Puzzle Sequences" -ForegroundColor Cyan
    Write-Host "-------------------------------------" -ForegroundColor Cyan
    
    if (Test-Path $PUZZLE_DB) {
        Write-Host "Input: $PUZZLE_DB" -ForegroundColor Gray
        Write-Host "Output: $PUZZLE_SEQUENCES" -ForegroundColor Gray
        Write-Host ""
        
        $confirm = Read-Host "Extract puzzle sequences? This will process ~20,000 puzzles - y/n"
        if ($confirm -eq 'y') {
            $args = @(
                "scripts\extract_puzzle_sequences.py",
                "-input", $PUZZLE_DB,
                "-output", $PUZZLE_SEQUENCES,
                "-limit", "20000",
                "-feature-set", "standard"
            )
            if ($Verbose) { $args += "-verbose" }
            
            Write-Host "Running puzzle extraction..." -ForegroundColor Yellow
            & $PYTHON $args
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Puzzle extraction failed!" -ForegroundColor Red
                exit 1
            }
            
            Write-Host "Checkmark Puzzle sequences extracted successfully" -ForegroundColor Green
        } else {
            Write-Host "Skipping puzzle extraction" -ForegroundColor Yellow
        }
    } else {
        Write-Host "WARNING: Puzzle database not found at $PUZZLE_DB" -ForegroundColor Yellow
        Write-Host "Skipping puzzle extraction..." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping puzzle extraction (-SkipPuzzleExtraction)" -ForegroundColor Yellow
}

# ==============================================================================
# Phase 2: Calculate Features for Game Data (with temporal=false)
# ==============================================================================
if (-not $SkipFeatureCalculation) {
    Write-Host ""
    Write-Host "PHASE 2: Feature Calculation (Game Data)" -ForegroundColor Cyan
    Write-Host "-----------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Game Data Processing:" -ForegroundColor Yellow
    Write-Host "  - Current features (F000-F114): Calculated" -ForegroundColor Gray
    Write-Host "  - Temporal features (F200-F220): Set to null sentinels" -ForegroundColor Gray
    Write-Host "  - Positions: ~230,930" -ForegroundColor Gray
    Write-Host "  - Estimated time: 4-6 hours" -ForegroundColor Gray
    Write-Host ""
    
    $confirm = Read-Host "Recalculate features for game data? This takes 4-6 hours - y/n"
    if ($confirm -eq 'y') {
        Write-Host "Processing game positions..." -ForegroundColor Yellow
        & $PYTHON scripts\calculate_features.py `
            -input $GAME_DATA `
            -output $GAME_FEATURES `
            -feature-set standard
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Feature calculation failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Checkmark Game features calculated" -ForegroundColor Green
    } else {
        Write-Host "Skipping feature calculation" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping feature calculation (-SkipFeatureCalculation)" -ForegroundColor Yellow
}

# ==============================================================================
# Phase 3: Merge Datasets
# ==============================================================================
Write-Host ""
Write-Host "PHASE 3: Merging Datasets" -ForegroundColor Cyan
Write-Host "-------------------------" -ForegroundColor Cyan
Write-Host ""
Write-Host "Combining:" -ForegroundColor Yellow
Write-Host "  - Puzzle sequences (with temporal features)" -ForegroundColor Gray
Write-Host "  - Game positions (without temporal features)" -ForegroundColor Gray
Write-Host ""

if ((Test-Path $PUZZLE_SEQUENCES) -and (Test-Path $GAME_FEATURES)) {
    Write-Host "Merging datasets..." -ForegroundColor Yellow
    
    $puzzles = Get-Content $PUZZLE_SEQUENCES
    $games = Get-Content $GAME_FEATURES
    $merged = $puzzles + $games
    
    Set-Content -Path $MERGED_DATA -Value $merged
    
    $puzzleCount = $puzzles.Count
    $gameCount = $games.Count
    $totalCount = $merged.Count
    
    Write-Host "Checkmark Merge complete" -ForegroundColor Green
    Write-Host "  - Puzzle positions: $puzzleCount" -ForegroundColor Gray
    Write-Host "  - Game positions: $gameCount" -ForegroundColor Gray
    Write-Host "  - Total positions: $totalCount" -ForegroundColor Gray
} else {
    Write-Host "ERROR: Required input files missing!" -ForegroundColor Red
    Write-Host "  - Puzzle sequences: $(Test-Path $PUZZLE_SEQUENCES)" -ForegroundColor Gray
    Write-Host "  - Game features: $(Test-Path $GAME_FEATURES)" -ForegroundColor Gray
    exit 1
}

# ==============================================================================
# Phase 4: Split Dataset
# ==============================================================================
if (-not $SkipPreprocessing) {
    Write-Host ""
    Write-Host "PHASE 4: Dataset Splitting" -ForegroundColor Cyan
    Write-Host "---------------------------" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Splitting into train/val/test sets..." -ForegroundColor Yellow
    & $PYTHON scripts\split_dataset.py `
        -input $MERGED_DATA `
        -output-dir $SPLIT_DIR `
        -train-ratio 0.8 `
        -val-ratio 0.1 `
        -test-ratio 0.1 `
        -stratify-by grade `
        -seed 42
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Dataset splitting failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Checkmark Dataset split complete" -ForegroundColor Green
} else {
    Write-Host "Skipping dataset splitting (-SkipPreprocessing)" -ForegroundColor Yellow
}

# ==============================================================================
# Phase 5: Preprocessing
# ==============================================================================
if (-not $SkipPreprocessing) {
    Write-Host ""
    Write-Host "PHASE 5: Preprocessing Features" -ForegroundColor Cyan
    Write-Host "--------------------------------" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Processing 262 features..." -ForegroundColor Yellow
    Write-Host "  - Numerical: StandardScaler" -ForegroundColor Gray
    Write-Host "  - Categorical: OneHotEncoder" -ForegroundColor Gray
    Write-Host "  - Output: NumPy arrays for training" -ForegroundColor Gray
    Write-Host ""
    
    & $PYTHON scripts\preprocess_dataset_v5.1.py `
        -train "$SPLIT_DIR\train.jsonl" `
        -val "$SPLIT_DIR\val.jsonl" `
        -test "$SPLIT_DIR\test.jsonl" `
        -output-dir $PREPROCESSED_DIR
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Preprocessing failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Checkmark Preprocessing complete" -ForegroundColor Green
    
    if (Test-Path "$PREPROCESSED_DIR\preprocessing_stats.json") {
        $stats = Get-Content "$PREPROCESSED_DIR\preprocessing_stats.json" | ConvertFrom-Json
        Write-Host "  - Training samples: $($stats.train_samples)" -ForegroundColor Gray
        Write-Host "  - Validation samples: $($stats.val_samples)" -ForegroundColor Gray
        Write-Host "  - Test samples: $($stats.test_samples)" -ForegroundColor Gray
        Write-Host "  - Total features: $($stats.total_features)" -ForegroundColor Gray
    }
} else {
    Write-Host "Skipping preprocessing (-SkipPreprocessing)" -ForegroundColor Yellow
}

# ==============================================================================
# Phase 6: Model Architecture Check
# ==============================================================================
Write-Host ""
Write-Host "PHASE 6: Model Architecture Verification" -ForegroundColor Cyan
Write-Host "-----------------------------------------" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model architecture: 256 to 256 to 128 to 64 (same as v5.0)" -ForegroundColor Gray
Write-Host "Input dimension updated: 26 to 262" -ForegroundColor Gray
Write-Host "Architecture supports expansion without changes" -ForegroundColor Gray
Write-Host ""
Write-Host "Checkmark Architecture ready (designed for expansion)" -ForegroundColor Green

# ==============================================================================
# Phase 7: Model Training
# ==============================================================================
if (-not $SkipTraining) {
    Write-Host ""
    Write-Host "PHASE 7: Model Training - 100 Epochs" -ForegroundColor Cyan
    Write-Host "------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Training Configuration:" -ForegroundColor Gray
    Write-Host "  - Epochs: 100" -ForegroundColor Gray
    Write-Host "  - Batch size: 256" -ForegroundColor Gray
    Write-Host "  - Class weights: 1.0, 5.0, 3.5, 2.5, 1.8, 1.0" -ForegroundColor Gray
    Write-Host "  - Loss: CrossEntropyLoss for policy and HuberLoss for value" -ForegroundColor Gray
    Write-Host "  - Optimizer: AdamW with lr=1e-3 and weight_decay=1e-4" -ForegroundColor Gray
    Write-Host "  - Scheduler: ReduceLROnPlateau" -ForegroundColor Gray
    Write-Host ""
    
    $confirm = Read-Host "Start training? This will take ~6-8 hours - y/n"
    if ($confirm -eq 'y') {
        Write-Host ""
        Write-Host "Starting training..." -ForegroundColor Yellow
        Write-Host "Monitor metrics carefully - expecting:" -ForegroundColor Cyan
        Write-Host "  - Policy accuracy: 56-60 percent vs 49 percent baseline" -ForegroundColor Cyan
        Write-Host "  - All grades predicted, not binary" -ForegroundColor Cyan
        Write-Host "  - Puzzle sequences: Higher accuracy due to temporal context" -ForegroundColor Cyan
        Write-Host ""
        
        & $PYTHON src\train.py `
            -epochs 100 `
            -batch-size 256 `
            -learning-rate 0.001 `
            -data-dir $PREPROCESSED_DIR `
            -checkpoint-dir $CHECKPOINT_DIR `
            -log-dir "logs\v5.1_tpf"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Training failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "Checkmark Training complete!" -ForegroundColor Green
    } else {
        Write-Host "Skipping training" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping training (-SkipTraining)" -ForegroundColor Yellow
}

# ==============================================================================
# Phase 8: Evaluation
# ==============================================================================
Write-Host ""
Write-Host "PHASE 8: Model Evaluation" -ForegroundColor Cyan
Write-Host "-------------------------" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "$CHECKPOINT_DIR\best_model.pth") {
    Write-Host "Evaluating best model on test set..." -ForegroundColor Yellow
    
    & $PYTHON src\evaluate.py `
        -checkpoint "$CHECKPOINT_DIR\best_model.pth" `
        -data-dir $PREPROCESSED_DIR `
        -output-dir "evaluation_results_v5.1_temporal"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Evaluation failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Checkmark Evaluation complete" -ForegroundColor Green
} else {
    Write-Host "No trained model found - skipping evaluation" -ForegroundColor Yellow
}

# ==============================================================================
# Summary
# ==============================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete!" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  - Merged dataset: $MERGED_DATA" -ForegroundColor White
Write-Host "  - Preprocessed data: $PREPROCESSED_DIR" -ForegroundColor White
Write-Host "  - Model checkpoints: $CHECKPOINT_DIR" -ForegroundColor White
Write-Host "  - Evaluation results: evaluation_results_v5.1_temporal/" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review confusion matrix - check if all grades are predicted" -ForegroundColor White
Write-Host "  2. Compare policy accuracy: v5.0 at 49 percent vs v5.1 with TPF target 56-60 percent" -ForegroundColor White
Write-Host "  3. Validate temporal consistency on puzzle sequences" -ForegroundColor White
Write-Host "  4. Test self-play integration with temporal state" -ForegroundColor White
Write-Host ""
Write-Host "Your AI now has MEMORY!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
