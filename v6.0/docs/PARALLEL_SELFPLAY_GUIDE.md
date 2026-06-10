# Parallel Self-Play Quick Reference
**V7P3R AI v6.1 - Fast Training Data Generation**

## Performance Comparison

### Sequential Mode (1 worker)
```bash
python scripts/stage2/run_batch_selfplay.py
```
- **Time**: ~9-10 hours
- **CPU**: 1 core utilized
- **Memory**: ~100-200MB
- **Use when**: Background generation overnight

### Parallel Mode (4 workers - RECOMMENDED)
```bash
python scripts/stage2/run_batch_selfplay.py --workers 4
```
- **Time**: ~2.5 hours ⚡
- **CPU**: 4 cores utilized
- **Memory**: ~400-800MB
- **Use when**: Active development, want results today

### Parallel Mode (8 workers - Maximum)
```bash
python scripts/stage2/run_batch_selfplay.py --workers 8
```
- **Time**: ~1.5 hours 🚀
- **CPU**: 8 cores utilized  
- **Memory**: ~800MB-1.5GB
- **Use when**: Powerful desktop, need results ASAP

## How It Works

**Multiprocessing Architecture**:
- Each worker is a separate Python process (avoids GIL)
- Each worker loads its own Stage 1 model copy (~20MB per worker)
- Games are distributed across workers using `Pool.imap_unordered()`
- No shared state = no locks or synchronization overhead
- Results are collected and saved as they complete

**Safety**:
- Resume capability works with parallel mode
- Progress saved every 10 games
- Crash-resistant: if one worker fails, others continue
- Output files naturally segregated by game_id (no conflicts)

## Optimal Worker Count

**Desktop (8-16 cores)**:
- Recommended: `--workers 4` or `--workers 6`
- Maximum: `--workers 8`
- Diminishing returns beyond 8 workers

**Laptop (4-8 cores)**:
- Recommended: `--workers 2` or `--workers 4`
- Leaves cores free for other tasks

**Workstation (16+ cores)**:
- Recommended: `--workers 8`
- Can go higher but self-play is already fast at 8

## Command Examples

### Basic parallel run (4 workers)
```bash
python scripts/stage2/run_batch_selfplay.py --workers 4
```

### Custom output directory
```bash
python scripts/stage2/run_batch_selfplay.py \
  --workers 4 \
  --output data/stage2/batch_284_fast
```

### Resume interrupted run
```bash
# Parallel mode automatically resumes from last checkpoint
python scripts/stage2/run_batch_selfplay.py --workers 4
# Will show: "✓ Resuming from game 127"
```

### Fresh start (no resume)
```bash
python scripts/stage2/run_batch_selfplay.py --workers 4 --no-resume
```

### Different game count (testing)
```bash
# Generate only 50 games for quick test
python scripts/stage2/run_batch_selfplay.py --workers 4 --games 50
```

## Resource Monitoring

**CPU Usage**:
- Sequential: ~12-15% (1 core on 8-core CPU)
- 4 workers: ~50% (4 cores on 8-core CPU)
- 8 workers: ~95-100% (all cores)

**Memory Usage** (typical):
- Base: ~100MB (Python + libraries)
- Per worker: ~100-150MB (model + game state)
- 4 workers: ~600-800MB total
- 8 workers: ~1.2-1.5GB total

**Disk I/O**:
- Minimal (JSONL writes are small)
- ~50-100KB per game
- Total: ~14-28MB for 284 games

## Expected Output

```
V7P3R AI Stage 2 Training Data Generation (PARALLEL)
======================================================================
Target games: 284
Workers: 4
Scenario distribution:
  blitz_early         :  68 games ( 23.9%)
  blitz_midgame       :  60 games ( 21.1%)
  blitz_endgame       :  42 games ( 14.8%)
  bullet_early        :  17 games (  6.0%)
  ...

Estimated completion time: ~2.5-3.0 hours
Mode: PARALLEL
Workers: 4

Starting parallel processing with 4 workers...
[1/284] Game 42 complete (blitz_midgame): 1-0 (31 moves)
[2/284] Game 17 complete (bullet_early): 0-1 (19 moves)
[3/284] Game 88 complete (blitz_endgame): 1/2-1/2 (47 moves)
...
```

## Troubleshooting

**"Workers must be >= 1"**:
- Don't use `--workers 0`, minimum is 1

**"Only N CPUs available"**:
- Warning only, will still run but may be slower
- Reduce worker count to match CPU count

**Slow performance with parallel**:
- Disk bottleneck (unlikely with JSONL)
- Memory pressure (reduce workers)
- Check CPU usage (should be near 100% with max workers)

**Games completing out of order**:
- Expected behavior! Workers finish at different times
- Game numbers in output won't be sequential
- Final dataset will have all 284 games

## Integration with Other Scripts

**After parallel generation**:
```bash
# 1. Verify data quality
python scripts/stage2/verify_compatibility.py \
  --data data/stage2/selfplay_batch_284

# 2. Train Stage 2 model
python scripts/stage2/train_stage2.py \
  --data data/stage2/selfplay_batch_284 \
  --epochs 30 \
  --batch-size 256

# 3. Test integrated engine
python scripts/engine/test_integrated_engine.py
```

## Performance Tips

1. **Close other apps**: Free up CPU for workers
2. **Use SSD**: Faster I/O for model loading and data saving
3. **Monitor progress**: Check output every 30 mins to ensure no stalls
4. **Start with 4 workers**: Good balance of speed vs resource usage
5. **Run overnight with fewer workers**: If you don't need results immediately

## Status Check

**Current implementation**:
- ✅ Multiprocessing.Pool support
- ✅ Resume capability with parallel
- ✅ Progress tracking per game
- ✅ Auto-detection of CPU count
- ✅ Worker validation and warnings
- ✅ Estimated completion time calculation

**Ready for production use!** 🚀
