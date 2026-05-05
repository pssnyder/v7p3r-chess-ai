# Containerized Training System - Quick Reference

Complete Docker-based training infrastructure for 48-hour unattended runs.

## 📦 What Was Created

### Core Infrastructure
- ✅ **Dockerfile** - Multi-stage build with PyTorch, CUDA, Stockfish
- ✅ **docker-compose.yml** - Orchestration with GPU support, TensorBoard
- ✅ **docker-compose.cpu.yml** - CPU fallback configuration
- ✅ **.dockerignore** - Build optimization (excludes data/models)

### Orchestration & Monitoring
- ✅ **scripts/48h_training_orchestrator.py** - Master controller for 4-phase training
- ✅ **scripts/training_health_check.py** - Container health monitoring
- ✅ **scripts/test_docker_setup.py** - Pre-flight validation

### Documentation & Utilities
- ✅ **DOCKER_TRAINING_GUIDE.md** - Complete setup and troubleshooting guide
- ✅ **start_docker_training.bat** - One-click Windows launcher
- ✅ **CONTAINERIZED_TRAINING_README.md** - This file

---

## 🚀 Quick Start (3 Steps)

### 1. Test Setup
```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"
python scripts/test_docker_setup.py
```

### 2. Launch Training (Easy Mode)
```powershell
.\start_docker_training.bat
```

### 3. Monitor Progress
- **TensorBoard**: http://localhost:6006
- **Logs**: `docker-compose logs -f training`
- **Health**: `docker-compose exec training python scripts/training_health_check.py`

---

## 🎯 Training Phases (48 Hours Total)

| Phase | Duration | Data Added | Output Model | Goal |
|-------|----------|------------|--------------|------|
| **1** | 12h | Endgame + 500K puzzles | v5.0 | Fix endgame weakness (7% → 20%) |
| **2** | 12h | 100K opening positions | v5.1 | Master-level opening theory |
| **3** | 12h | 100K master games | v5.2 | Learn 2200-2800 ELO patterns |
| **4** | 12h | 50K positional patterns | v5.3 | Strategic understanding |

**Final Output**: `models/phase4_positional/best_checkpoint.pt` (v5.3)

---

## 🔄 Auto-Recovery Features

✅ **Checkpoint Resume** - Restarts from last epoch if container crashes  
✅ **Failure Retry** - 3 attempts per phase with exponential backoff  
✅ **Health Monitoring** - Auto-detects stuck training (5-min intervals)  
✅ **Graceful Shutdown** - SIGTERM handling preserves state  
✅ **Volume Persistence** - All progress saved to host filesystem  

**Recovery Test**:
```powershell
# Simulate crash
docker-compose kill training

# Auto-resume
docker-compose up -d
# ↑ Picks up exactly where it left off
```

---

## 📊 Monitoring & Debugging

### Real-Time Metrics
```powershell
# TensorBoard (browser)
http://localhost:6006

# Live logs
docker-compose logs -f training

# Health check
docker-compose exec training python scripts/training_health_check.py

# Current phase
docker-compose exec training cat checkpoints/orchestrator_state.json
```

### Common Issues

**Container won't start?**
```powershell
docker-compose logs training
# Check for port conflicts, GPU driver, volume permissions
```

**Training stuck?**
```powershell
docker-compose restart training
# Health check auto-runs every 5min
```

**Out of memory?**
```powershell
# Edit docker-compose.yml
mem_limit: 48g  # Increase from 32g
```

---

## 🛑 Stop & Cleanup

```powershell
# Graceful stop (saves progress)
docker-compose down

# Force stop (emergency)
docker-compose kill

# Complete cleanup (⚠️ deletes checkpoints!)
docker-compose down -v
```

---

## 📁 File Structure

```
v4.0/
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # GPU orchestration
├── docker-compose.cpu.yml              # CPU fallback
├── .dockerignore                       # Build optimization
├── start_docker_training.bat           # Windows launcher
├── DOCKER_TRAINING_GUIDE.md            # Full documentation
├── CONTAINERIZED_TRAINING_README.md    # This file
│
├── scripts/
│   ├── 48h_training_orchestrator.py   # Master controller
│   ├── training_health_check.py       # Health monitoring
│   ├── test_docker_setup.py           # Pre-flight checks
│   └── train_move_ordering.py         # Training script (existing)
│
└── [Mounted Volumes - Persists Across Restarts]
    ├── data/                           # Training datasets
    ├── models/                         # Model checkpoints
    ├── logs/                           # Text logs
    ├── tensorboard_logs/               # TensorBoard metrics
    └── checkpoints/                    # Orchestrator state
```

---

## ✅ Pre-Flight Checklist

Before leaving for the weekend:

- [ ] Docker Desktop running
- [ ] GPU drivers up to date (if using GPU)
- [ ] 50+ GB free disk space
- [ ] Test setup passed: `python scripts/test_docker_setup.py`
- [ ] TensorBoard accessible: http://localhost:6006
- [ ] Health check works: `docker-compose exec training python scripts/training_health_check.py`
- [ ] Current models backed up:
  ```powershell
  xcopy /E /I models models_backup_20260430
  ```

---

## 🎓 Expected Results

**After 48 Hours**:
- ✅ 4 model checkpoints (v5.0, v5.1, v5.2, v5.3)
- ✅ Complete TensorBoard metrics
- ✅ Training logs for each phase
- ✅ Final model: `models/phase4_positional/best_checkpoint.pt`

**Success Criteria**:
- Top-5 accuracy: ≥97% (maintain current)
- Training loss: <0.15 (improve from 0.18)
- All 4 phases completed
- No critical errors in health checks

**Next Steps After Training**:
1. Verify all phases completed: `cat checkpoints/orchestrator_state.json`
2. Run A/B test: v5.3 vs v4.0 (50 games)
3. Test eval alignment: Material/Positional/v18.3 variants should produce DIFFERENT games
4. If successful: Deploy v5.3 to production

---

## 🔗 Related Files

- **Full Guide**: [DOCKER_TRAINING_GUIDE.md](DOCKER_TRAINING_GUIDE.md)
- **Training Plan**: Session memory `/memories/session/plan.md`
- **Existing Scripts**: `scripts/train_move_ordering.py`, `scripts/preprocess_puzzles_with_stockfish.py`
- **Model Architecture**: `src/models/move_ordering_network.py`

---

## 💡 Pro Tips

1. **Start with test run**: Use `--duration-hours 1` to test orchestrator before weekend
2. **Monitor first hour**: Check TensorBoard and logs to verify training is progressing
3. **GPU temperature**: Monitor `nvidia-smi` - if >85°C, reduce power limit
4. **Disk space**: Training generates ~5GB logs + ~2GB checkpoints
5. **Network not required**: All training is local (no cloud dependencies)

---

## 🚨 Emergency Contacts

**If something goes wrong**:
1. Check health: `docker-compose exec training python scripts/training_health_check.py`
2. View logs: `docker-compose logs --tail 100 training`
3. Generate diagnostic dump:
   ```powershell
   docker-compose exec training bash -c "
     nvidia-smi && df -h && free -h && grep ERROR logs/*.log
   " > emergency_diagnostic.txt
   ```

**Recovery procedure**:
```powershell
# Stop everything
docker-compose down

# Check state
cat checkpoints/orchestrator_state.json

# Resume from last checkpoint
docker-compose up -d
```

---

**Created**: April 30, 2026  
**Version**: 1.0  
**Status**: Ready for weekend training run ✅
