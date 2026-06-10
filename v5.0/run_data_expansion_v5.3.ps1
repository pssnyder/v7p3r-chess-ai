# V7P3R AI v5.3 - Multi-Engine Data Expansion Pipeline
# Orchestrates data collection from multiple sources

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "V7P3R AI v5.3 - MULTI-ENGINE DATA EXPANSION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Data Sources:" -ForegroundColor Yellow
Write-Host "  1. V7P3R Engine Puzzle Solving (18.4, 18.3, 18.0, 17.1)" -ForegroundColor White
Write-Host "  2. Lichess 4M Puzzle Database (full extraction)" -ForegroundColor White
Write-Host "  3. C0BR4 Game History (10k games)" -ForegroundColor White
Write-Host "  4. V7P3R Bot Unrated Games (historical)" -ForegroundColor White
Write-Host ""

Write-Host "Target Dataset Size: 600k-750k positions" -ForegroundColor Green
Write-Host "Focus Metric: Good Move Rate (% predictions in grades 0-2)" -ForegroundColor Green
Write-Host ""

$startTime = Get-Date

# Phase 1: Multi-Engine Puzzle Solving
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 1: Multi-Engine Puzzle Solving" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Running 4 V7P3R engines through 10k puzzles each..." -ForegroundColor Yellow
Write-Host "This will take approximately 3-4 hours (total: 40k positions)" -ForegroundColor White
Write-Host ""

# Ask user if they want to proceed
$proceed = Read-Host "Start multi-engine puzzle solving? (y/n)"

if ($proceed -eq 'y') {
    python scripts\multi_engine_puzzle_solver.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Multi-engine puzzle solving completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Multi-engine puzzle solving failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipped multi-engine puzzle solving" -ForegroundColor Yellow
}

# Phase 2: Full Lichess Puzzle Extraction
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 2: Full Lichess Puzzle Database Extraction" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Extracting 100k+ puzzles from 4M Lichess database..." -ForegroundColor Yellow
Write-Host "This will take approximately 2-3 hours" -ForegroundColor White
Write-Host ""

$proceed = Read-Host "Start full puzzle extraction? (y/n)"

if ($proceed -eq 'y') {
    python scripts\extract_puzzle_sequences.py `
        --input "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\csv_data_puzzles\lichess_db_puzzle.csv" `
        --output data\puzzles\puzzle_sequences_full.jsonl `
        --num-puzzles 100000 `
        --rating-min 1500 `
        --rating-max 2500
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Full puzzle extraction completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Puzzle extraction failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipped full puzzle extraction" -ForegroundColor Yellow
}

# Phase 3: C0BR4 Game Data Integration
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 3: C0BR4 Game Data Integration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Processing C0BR4 (cousin engine) game history..." -ForegroundColor Yellow
Write-Host "Expected: 10k games → ~200k positions" -ForegroundColor White
Write-Host ""

$proceed = Read-Host "Start C0BR4 data integration? (y/n)"

if ($proceed -eq 'y') {
    # Check if C0BR4 integration script exists
    if (Test-Path "scripts\integrate_cobra_games.py") {
        python scripts\integrate_cobra_games.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ C0BR4 data integration completed!" -ForegroundColor Green
        } else {
            Write-Host "❌ C0BR4 integration failed!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "⚠️  C0BR4 integration script not found - create manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "⏭️  Skipped C0BR4 integration" -ForegroundColor Yellow
}

# Phase 4: Merge All Datasets
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 4: Dataset Merging and Preprocessing" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Merging all data sources..." -ForegroundColor Yellow
Write-Host ""

$proceed = Read-Host "Start dataset merging? (y/n)"

if ($proceed -eq 'y') {
    # Merge datasets
    python scripts\merge_datasets_v5.3.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dataset merging completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Dataset merging failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipped dataset merging" -ForegroundColor Yellow
}

# Phase 5: Oversample Rare Grades
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 5: Balance Grade Distribution (Oversampling)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Oversampling rare grades (1, 3, 4) to balance distribution..." -ForegroundColor Yellow
Write-Host ""

$proceed = Read-Host "Start grade balancing? (y/n)"

if ($proceed -eq 'y') {
    python scripts\balance_grade_distribution.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Grade balancing completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Grade balancing failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipped grade balancing" -ForegroundColor Yellow
}

# Phase 6: Preprocess
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PHASE 6: Feature Preprocessing" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$proceed = Read-Host "Start preprocessing? (y/n)"

if ($proceed -eq 'y') {
    python scripts\preprocess_dataset_v5.1.py `
        --input data\final\v7p3r_ai_v5.3_merged.jsonl `
        --output data\preprocessed_v5.3 `
        --version v5.3
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Preprocessing completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Preprocessing failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipped preprocessing" -ForegroundColor Yellow
}

# Summary
$endTime = Get-Date
$duration = $endTime - $startTime
$hours = [math]::Floor($duration.TotalHours)
$minutes = $duration.Minutes

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DATA EXPANSION PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Duration: $hours hours $minutes minutes" -ForegroundColor White
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Train v5.3 with expanded dataset" -ForegroundColor White
Write-Host "  2. Evaluate with Good Move Metrics (scripts/good_move_metrics.py)" -ForegroundColor White
Write-Host "  3. Target: >70% Good Move Rate (grades 0-2)" -ForegroundColor White
Write-Host ""
Write-Host "Training Command:" -ForegroundColor Yellow
Write-Host "  python src\train.py --config configs\training_config_v5.3_expanded.yaml --data-dir data\preprocessed_v5.3" -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
