# Docker Training Guide - V7P3R AI

Complete guide for running 48-hour unattended training in Docker containers.

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   ```powershell
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **NVIDIA Docker Runtime** (for GPU training)
   ```powershell
   # Windows: Install NVIDIA Container Toolkit
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   
   # Verify GPU access
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **System Requirements**
   - RAM: 16GB minimum, 32GB recommended
   - Disk: 50GB free space for data + models
   - GPU: NVIDIA GPU with 8GB+ VRAM (optional, CPU fallback available)

### Build and Start

```powershell
# Navigate to v4.0 directory
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

# Build Docker image (first time, ~5-10 minutes)
docker-compose build

# Start 48-hour training (GPU)
docker-compose up -d

# OR: Start CPU-only training
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Monitor Progress

```powershell
# View live logs
docker-compose logs -f training

# Check health status
docker-compose exec training python scripts/training_health_check.py

# Access TensorBoard (in browser)
http://localhost:6006

# View orchestrator state
docker-compose exec training cat checkpoints/orchestrator_state.json
```

---

## 📊 Training Phases

The orchestrator automatically progresses through 4 phases:

### Phase 1: Endgame Expansion (12 hours)
- **Data**: 52,500 endgame positions + 500,000 themed puzzles
- **Output**: `models/phase1_endgame_puzzles/best_checkpoint.pt` (v5.0)
- **Goal**: Increase endgame representation from 7% → 20%

### Phase 2: Opening Theory (12 hours)
- **Data**: Phase 1 + 100,000 opening positions from master games
- **Output**: `models/phase2_opening_theory/best_checkpoint.pt` (v5.1)
- **Goal**: Replace V7P3R opening patterns with master-level theory

### Phase 3: Master Games (12 hours)
- **Data**: Phase 2 + 100,000 positions from 2200-2800 ELO games
- **Output**: `models/phase3_master_games/best_checkpoint.pt` (v5.2)
- **Goal**: Learn "what should be done" vs "what V7P3R does"

### Phase 4: Positional Refinement (12 hours)
- **Data**: Phase 3 + 50,000 quiet positional moves
- **Output**: `models/phase4_positional/best_checkpoint.pt` (v5.3)
- **Goal**: Improve strategic understanding and move diversity

---

## 🔄 Auto-Recovery Features

### Checkpoint System
- **Frequency**: Every epoch (50 epochs per phase)
- **Storage**: `models/<phase_name>/latest_checkpoint.pt`
- **Best Model**: `models/<phase_name>/best_checkpoint.pt`
- **State File**: `checkpoints/orchestrator_state.json`

### Failure Scenarios Handled

1. **Container Crash**: Restart from last checkpoint
   ```powershell
   docker-compose restart training
   # Automatically resumes from last epoch
   ```

2. **Out of Memory**: Graceful degradation
   - Reduces batch size automatically
   - Enables gradient accumulation
   - Retries with exponential backoff

3. **Network Interruption**: Local persistence
   - All data in mounted volumes
   - No cloud dependencies
   - Offline-capable

4. **Power Outage**: Resume on reboot
   ```powershell
   # After system restart
   docker-compose up -d
   # Picks up where it left off
   ```

### Health Checks

Runs every 5 minutes, checks:
- ✅ Checkpoint modified within last 30 minutes
- ✅ No critical errors in logs
- ✅ Memory usage < 95%
- ✅ Orchestrator state valid

**Manual Health Check**:
```powershell
docker-compose exec training python scripts/training_health_check.py
```

---

## 📁 Volume Mounts (Persistent Storage)

All progress persists across container restarts:

```
Host Path                    → Container Path              Purpose
./data                      → /workspace/data            Training datasets
./models                    → /workspace/models          Model checkpoints
./logs                      → /workspace/logs            Text logs
./tensorboard_logs          → /workspace/tensorboard_logs TensorBoard metrics
```

**Backup Before Training**:
```powershell
# Backup current models
cp -r models models_backup_$(Get-Date -Format "yyyyMMdd_HHmmss")
```

---

## 🎛️ Configuration

### Environment Variables (docker-compose.yml)

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0        # GPU index (0, 1, 2...) or "" for CPU
  - TRAINING_DURATION=48h         # Total duration
  - AUTO_RECOVER=true             # Enable checkpoint resume
  - ENABLE_TENSORBOARD=true       # Start TensorBoard server
  - LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
  - OMP_NUM_THREADS=8             # CPU threads for PyTorch
```

### Phase Customization

Edit `scripts/48h_training_orchestrator.py`:

```python
self.phases = [
    {
        'name': 'phase1_endgame_puzzles',
        'epochs': 50,              # Increase for more training
        'duration_hours': 12,      # Time budget
        # ...
    },
    # ...
]
```

### Resource Limits

Edit `docker-compose.yml`:

```yaml
mem_limit: 32g              # Max RAM
shm_size: 8g                # Shared memory for data loading
```

---

## 🛑 Stopping and Cleanup

### Graceful Stop (preserves checkpoints)
```powershell
# Stop training, save current progress
docker-compose down

# Checkpoints and logs are preserved in mounted volumes
```

### Force Stop (emergency)
```powershell
# Kill container immediately
docker-compose kill training

# Restart from last checkpoint
docker-compose up -d
```

### Complete Cleanup (WARNING: Deletes all progress)
```powershell
# Stop and remove containers
docker-compose down

# Remove volumes (CAUTION: Deletes checkpoints!)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

---

## 📈 Monitoring and Debugging

### TensorBoard

1. **Access**: http://localhost:6006
2. **Metrics**:
   - Training/validation loss
   - Top-5/Top-10 accuracy
   - Learning rate
   - GPU utilization (if available)

### Log Files

```powershell
# Orchestrator log (high-level phase progress)
docker-compose exec training tail -f logs/orchestrator.log

# Training log (detailed epoch metrics)
docker-compose exec training tail -f logs/training.log

# Combined view
docker-compose logs -f training
```

### Check Current Phase

```powershell
docker-compose exec training python -c "
import json
with open('checkpoints/orchestrator_state.json') as f:
    state = json.load(f)
    print(f'Current Phase: {state[\"current_phase\"]}')
    print(f'Completed: {state[\"completed_phases\"]}')
"
```

### GPU Monitoring

```powershell
# Inside container
docker-compose exec training nvidia-smi

# Watch GPU usage
docker-compose exec training watch -n 1 nvidia-smi
```

---

## 🐛 Troubleshooting

### Container Won't Start

**Symptom**: `docker-compose up` fails
```powershell
# Check logs
docker-compose logs training

# Common fixes:
# 1. Port conflict (TensorBoard on 6006)
docker ps  # Check if port 6006 is in use

# 2. GPU not available
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d

# 3. Volume permissions
docker-compose down -v
docker-compose up -d
```

### Training Stuck

**Symptom**: No progress for >30 minutes
```powershell
# Check health
docker-compose exec training python scripts/training_health_check.py

# Check GPU usage
docker-compose exec training nvidia-smi

# Restart with checkpoint resume
docker-compose restart training
```

### Out of Memory

**Symptom**: Container killed, exit code 137
```powershell
# Edit docker-compose.yml: Reduce batch size
# In orchestrator.py, change:
'--batch-size', '64',  # Reduce from 128

# Or: Increase memory limit
mem_limit: 48g  # Increase from 32g
```

### Data Not Found

**Symptom**: "Data source not found" errors
```powershell
# Check data directory
docker-compose exec training ls -la data/

# Re-run data extraction
docker-compose exec training python scripts/extract_endgame_positions.py
```

---

## 🔬 Advanced Usage

### Run Single Phase

```powershell
# Run only Phase 1
docker-compose exec training python -c "
from scripts.48h_training_orchestrator import TrainingOrchestrator
orch = TrainingOrchestrator(auto_recover=True)
orch.run_training_phase(orch.phases[0], orch.load_state())
"
```

### Custom Training Script

```powershell
# Run custom training instead of orchestrator
docker-compose run training python scripts/train_move_ordering.py \
  --data-path data/custom_dataset.json \
  --checkpoint-dir models/custom_model \
  --num-epochs 100
```

### Export Final Model

```powershell
# Copy best checkpoint to host
docker cp v7p3r-training:/workspace/models/phase4_positional/best_checkpoint.pt ./v5.3_final.pt

# Or: Use volume mount
cp models/phase4_positional/best_checkpoint.pt ../v5.3_final.pt
```

---

## 📝 Pre-Flight Checklist

Before starting 48-hour training:

- [ ] Docker Desktop running
- [ ] NVIDIA Docker installed (for GPU)
- [ ] At least 50GB free disk space
- [ ] Training data prepared (`data/` directory)
- [ ] Previous models backed up
- [ ] TensorBoard accessible (test http://localhost:6006)
- [ ] Health check script works:
  ```powershell
  docker-compose exec training python scripts/training_health_check.py
  ```
- [ ] Orchestrator state clean or ready to resume:
  ```powershell
  cat checkpoints/orchestrator_state.json
  ```

---

## 🎯 Expected Timeline

**Total Duration**: 48 hours

```
Phase 1 (Endgame + Puzzles):    0h → 12h   (25%)
Phase 2 (Opening Theory):      12h → 24h   (50%)
Phase 3 (Master Games):        24h → 36h   (75%)
Phase 4 (Positional):          36h → 48h   (100%)
```

**Checkpoints Saved**:
- Every epoch (50 times per phase = 200 total)
- Best model per phase (4 total)
- Latest model (rolling, 1 per phase)

**Expected Output**:
- `models/phase4_positional/best_checkpoint.pt` - Final v5.3 model
- `tensorboard_logs/` - Complete training metrics
- `logs/` - Detailed execution logs

---

## 🚨 Emergency Procedures

### Training Diverging (Loss Increasing)

```powershell
# Stop training
docker-compose down

# Reduce learning rate
# Edit scripts/48h_training_orchestrator.py:
'--learning-rate', '1e-4',  # Reduce from 5e-4

# Resume
docker-compose up -d
```

### System Overheating

```powershell
# Reduce GPU usage
docker-compose exec training nvidia-smi -pl 200  # Limit power to 200W

# Or: Switch to CPU
docker-compose down
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Disk Full

```powershell
# Free space
docker system prune -a  # Remove unused images/containers

# Clean old checkpoints (keep only best)
docker-compose exec training bash -c "
cd models/phase1_endgame_puzzles
ls -t *.pt | tail -n +2 | xargs rm
"
```

---

## 📞 Support

**Logs for Debugging**:
```powershell
# Full diagnostic dump
docker-compose exec training bash -c "
  echo '=== System Info ===' &&
  nvidia-smi &&
  echo '=== Disk Usage ===' &&
  df -h &&
  echo '=== Memory Usage ===' &&
  free -h &&
  echo '=== Recent Errors ===' &&
  grep ERROR logs/*.log | tail -20
" > diagnostic_dump.txt
```

**Health Report**:
```powershell
docker-compose exec training python scripts/training_health_check.py > health_report.txt
```

---

## ✅ Post-Training Validation

After 48 hours complete:

1. **Verify All Phases Completed**
   ```powershell
   docker-compose exec training cat checkpoints/orchestrator_state.json
   # Should show all 4 phases in completed_phases
   ```

2. **Check Final Model**
   ```powershell
   ls -lh models/phase4_positional/best_checkpoint.pt
   # Should be ~100MB
   ```

3. **Review TensorBoard**
   - Loss should be decreasing across all phases
   - Top-5 accuracy should be >97%
   - No sudden spikes or divergence

4. **Run A/B Test**
   ```powershell
   # Test v5.3 vs v4.0
   python test_ab_variants.py --model-v5 models/phase4_positional/best_checkpoint.pt
   ```

5. **Deploy to Production**
   - Copy `models/phase4_positional/best_checkpoint.pt` to `v4.0/models/stage2_combined/best_checkpoint.pt`
   - Update version in `v7p3r_v20_hybrid.py` to v20.1
   - Run tournament testing

---

## 📚 Additional Resources

- [PyTorch Docker Documentation](https://github.com/pytorch/pytorch#docker-image)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)
- [V7P3R Training Plan](docs/TRAINING_EXPANSION_PLAN.md) (if exists)

---

**Last Updated**: April 30, 2026  
**Version**: 1.0  
**Author**: V7P3R Development Team
