# V7P3R v20.0.2 A/B Test Variants - Setup & Testing Guide

## 🎯 Purpose

Test whether **simpler evaluation functions** work better with AI move ordering than v7p3r's complex evaluation.

## 📦 What Was Created

### 3 Engine Variants (All use AI ordering + v18.3 search)

| Variant | File | Evaluation | Hypothesis |
|---------|------|------------|------------|
| **v20.0.2** (Control) | `V7P3R_v20_Beta.bat` | v18.3 complex (PST + Material + 12+ bonuses) | Original hybrid architecture |
| **v20.0.2-Material** | `V7P3R_v20_Material.bat` | MaterialOpponent simple (material + bishop pair) | Simple eval = better AI alignment |
| **v20.0.2-Positional** | `V7P3R_v20_Positional.bat` | PositionalOpponent pure PST (no material) | PST-only = better positional AI |

### Supporting Files

- `v7p3r_v20_material_hybrid.py` - Material variant engine code
- `v7p3r_v20_positional_hybrid.py` - Positional variant engine code
- `v7p3r_v20_material_uci.py` - Material variant UCI interface
- `v7p3r_v20_positional_uci.py` - Positional variant UCI interface
- `test_ab_variants.py` - Quick verification test
- `OPPONENT_ENGINE_ANALYSIS.md` - Full tournament data analysis

---

## 🧪 Quick Verification Test

**Before tournament testing**, verify both variants work:

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"
python test_ab_variants.py
```

**Expected output:**
- Both engines initialize successfully
- Evaluation scores for test positions
- Search completes with TT/killer hits
- NPS comparison between variants

**What to look for:**
- **Material variant:** Should have FAST evaluation (~0.1ms)
- **Positional variant:** Should have moderate evaluation (~0.3ms)
- **Both:** Should use TT, killers, quiescence (same search as v20.0.2)

---

## 🏆 Tournament Setup (Arena GUI)

### Step 1: Add Engines to Arena

1. Open Arena → Engines → Install New Engine
2. Browse to: `e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0\`
3. Add these 3 engines:
   - `V7P3R_v20_Beta.bat` (name: **V7P3R v20.0.2 Beta**)
   - `V7P3R_v20_Material.bat` (name: **V7P3R v20.0.2-Material**)
   - `V7P3R_v20_Positional.bat` (name: **V7P3R v20.0.2-Positional**)

### Step 2: Create Tournament

**Tournament → New Tournament**

**Participants (6 engines):**
1. V7P3R v20.0.2 Beta (control)
2. V7P3R v20.0.2-Material (simple eval)
3. V7P3R v20.0.2-Positional (PST eval)
4. V7P3R_v18.3 (proven baseline)
5. MaterialOpponent (reference)
6. PositionalOpponent (reference)

**Settings:**
- **Type:** Round-robin (each pair plays)
- **Games per pairing:** 20 games minimum
- **Time control:** 5+3 blitz (gives AI time to order)
- **Opening book:** Disabled (test raw engine strength)
- **Adjudication:** 50-move draw, 3-fold repetition

**Total games:** 15 pairings × 20 games = **300 games**

### Step 3: Run Tournament

- Start tournament and let run overnight
- Monitor for crashes (check Arena logs)
- Watch first few games of each variant

---

## 📊 Metrics to Track

### Primary Metrics (Determine Winner)

| Metric | Goal | How to Measure |
|--------|------|----------------|
| **Win Rate vs Control** | ≥55% | Tournament results table |
| **Tactical Accuracy** | ≥15% (up from 10%) | Manual analysis of tactics |
| **Speed (NPS)** | ≥8,000 average | Arena "Info" panel during games |
| **Stability** | No crashes | Arena log file |

### Secondary Metrics (Understanding)

| Metric | Purpose |
|--------|---------|
| **Win % vs v18.3** | Does hybrid beat pure v18.3? |
| **Win % vs Opponent engines** | Does simplification help? |
| **Game phase accuracy** | Opening/middlegame/endgame decisions |
| **Move time distribution** | Consistent timing vs timeouts |
| **Blunder rate** | Material losses, hanging pieces |

---

## 🎯 Expected Outcomes

### Scenario 1: Material Variant Wins

**Result:** v20.0.2-Material scores ≥55% vs v20.0.2 control

**Interpretation:**
- AI ordering works BETTER with simple material evaluation
- Complex v18.3 eval was confusing the AI's move priorities
- AI's 100K tactical puzzle training aligns with material focus

**Next Steps:**
1. Deploy v20.0.2-Material as new hybrid baseline
2. Run extended testing (100 games vs v19.5 production)
3. Fine-tune AI on tactical puzzles (Stage 3 training)
4. Focus future training on material-winning patterns

---

### Scenario 2: Positional Variant Wins

**Result:** v20.0.2-Positional scores ≥55% vs v20.0.2 control

**Interpretation:**
- AI ordering works BETTER with pure PST evaluation
- AI's 374K game position training aligns with positional focus
- No material constants = clearer positional objectives

**Next Steps:**
1. Deploy v20.0.2-Positional as new hybrid baseline
2. Run extended testing (100 games vs v19.5 production)
3. Train AI on positional game records (Stage 3)
4. Emphasize piece coordination in training

---

### Scenario 3: v20.0.2 (Complex Eval) Still Best

**Result:** Neither variant beats v20.0.2 control

**Interpretation:**
- v18.3's complex evaluation IS working with AI ordering
- 10% tactical accuracy is an AI model limitation, not eval issue
- Need to improve AI model, not simplify evaluation

**Next Steps:**
1. Keep v20.0.2 (complex eval) architecture
2. Focus on AI model improvement (Stage 3 training)
3. Add tactical puzzle dataset to training
4. Consider different model architecture (transformers?)

---

### Scenario 4: All Hybrids Lose to v18.3

**Result:** v18.3 (no AI ordering) beats all AI variants

**Interpretation:**
- AI ordering overhead not worth the benefit yet
- 3ms AI overhead + evaluation mismatch > no AI
- Need significantly better AI model before retry

**Next Steps:**
1. Revert to pure v18.3 for production
2. Major AI model architecture redesign
3. Consider ensemble approaches
4. Research latest chess AI papers

---

## 📈 Post-Tournament Analysis

### Arena Tournament Results Location

Results saved to:
```
C:\Program Files (x86)\Arena\Tournaments\V7P3R_AB_Test_YYYYMMDD.txt
```

### Key Data to Extract

1. **Cross-table scores:**
   - Each variant vs each opponent
   - Win/Loss/Draw breakdown
   - Sonneborn-Berger tiebreak scores

2. **Individual game analysis:**
   - Open PGN file in Arena
   - Filter games by engine
   - Look for tactical blunders
   - Check endgame conversions

3. **Time management:**
   - Check for timeouts (time forfeit losses)
   - Average move time per variant
   - Time usage in critical positions

### Manual Tactical Analysis

**Test positions for each variant:**

```python
# Use test_v20_beta.py tactical suite
# Run for each variant, compare accuracy
python test_v20_beta.py  # v20.0.2 control
# Then modify script to load Material/Positional variants
```

**Expected tactical performance:**

| Variant | Tactical Accuracy | Reasoning |
|---------|-------------------|-----------|
| v20.0.2 | 10% (baseline) | Complex eval, AI trained on V7P3R |
| Material | **15-20%** (predicted) | Simple material focus, aligned with puzzles |
| Positional | 10-12% (predicted) | PST focus, not tactical |

---

## 📋 Success Criteria Checklist

### For Material Variant to Win

- [ ] Win rate ≥55% vs v20.0.2 control (20 game minimum)
- [ ] Tactical accuracy ≥15% on test suite (up from 10%)
- [ ] Average NPS ≥12,000 (faster than v20.0.2's 4,227)
- [ ] No crashes or illegal moves in 300 game tournament
- [ ] Win rate ≥50% vs v19.5 (production validation)

### For Positional Variant to Win

- [ ] Win rate ≥55% vs v20.0.2 control (20 game minimum)
- [ ] Game phase accuracy ≥70% (opening/middlegame/endgame)
- [ ] Positional blunders ≤5 per 100 games (hanging pieces)
- [ ] Average NPS ≥10,000 (PST evaluation overhead acceptable)
- [ ] No crashes or illegal moves in 300 game tournament
- [ ] Win rate ≥50% vs v19.5 (production validation)

### For Deployment (Either Variant)

**All criteria must pass:**

1. **Tournament Performance:**
   - ✅ Best performer in 300-game round-robin
   - ✅ Win rate ≥55% vs v20.0.2 control
   - ✅ Win rate ≥50% vs v18.3 (hybrid better than pure)

2. **Stability:**
   - ✅ Zero crashes in 300 games
   - ✅ No illegal move attempts
   - ✅ Consistent time management (no timeouts)

3. **Production Validation:**
   - ✅ Win rate ≥50% vs v19.5 current production (50 game minimum)
   - ✅ Performance consistent across time controls (1+1, 5+3, 15+10)

4. **Documentation:**
   - ✅ Tournament results documented
   - ✅ Evaluation differences explained
   - ✅ Deployment guide created

---

## 🚀 Deployment Process (If Variant Wins)

### If Material Variant Wins

1. **Rename files:**
   ```powershell
   # Backup current v20.0.2
   Copy-Item v7p3r_v20_hybrid.py v7p3r_v20_hybrid_BACKUP_complex_eval.py
   
   # Promote Material variant to main
   Copy-Item v7p3r_v20_material_hybrid.py v7p3r_v20_hybrid.py
   Copy-Item v7p3r_v20_material_uci.py v7p3r_v20_uci.py
   ```

2. **Update version string:**
   ```python
   # In v7p3r_v20_uci.py
   print("id name V7P3R v20.1.0 Beta (Hybrid AI + Material Eval)")
   ```

3. **Update CHANGELOG:**
   - Document v20.1.0 as Material eval upgrade
   - Reference A/B test tournament results
   - Note performance improvements

### If Positional Variant Wins

1. **Rename files:** (same as above, but Positional variant)
2. **Update version string:** `v20.1.0 Beta (Hybrid AI + PST Eval)`
3. **Update CHANGELOG:** Document PST eval upgrade

---

## 🔬 Future Research Paths

### If Material Wins (Tactical Focus)

**Stage 3 Training:**
- Add tactical puzzle dataset (Lichess puzzles DB)
- Fine-tune on material-winning patterns
- Increase puzzle weight in training mix

**Model Architecture:**
- Add tactical theme attention heads
- Separate tactical vs positional pathways
- Ensemble: tactical model + positional model

### If Positional Wins (Positional Focus)

**Stage 3 Training:**
- Add grandmaster game database
- Fine-tune on positional game records
- Emphasize piece coordination patterns

**Model Architecture:**
- Add positional feature extractors
- Long-range attention for piece coordination
- Game phase-aware model (opening/middle/endgame)

### If Complex Eval Wins (Model Improvement)

**Stage 3 Training:**
- Increase training dataset size (1M+ positions)
- Add evaluation score targets (not just move ordering)
- Multi-task learning (move order + position eval)

**Model Architecture:**
- Transformer architecture (replace LSTM)
- Self-attention for piece relationships
- Larger model (10M+ parameters if GPU available)

---

## 📞 Support & Troubleshooting

### Common Issues

**Problem:** Material variant crashes on startup

**Solution:**
```powershell
# Check model path exists
python -c "from pathlib import Path; print(Path('models/stage2_combined/best_checkpoint.pt').exists())"

# If False, model not found - verify model file location
```

---

**Problem:** Positional variant plays slowly (NPS < 1000)

**Solution:**
- PST evaluation is O(64) per position (64 squares)
- Expected NPS: 8,000-12,000 (slower than Material's 15,000+)
- If NPS < 5,000, check for infinite loops in evaluation

---

**Problem:** Arena says "engine not responding"

**Solution:**
- Check BAT file path is correct
- Verify Python is in PATH
- Run BAT file manually to see error messages
- Check UCI protocol output (should print "uciok" on startup)

---

## 📝 Results Template

**After tournament completes, fill this in:**

```markdown
# V7P3R v20.0.2 A/B Test Results

**Date:** [Date]
**Games Played:** [X] of 300
**Time Control:** 5+3 blitz

## Final Standings

| Rank | Engine | Score | Win % | Notes |
|------|--------|-------|-------|-------|
| 1 | [Winner] | X/Y | Z% | [Observations] |
| 2 | [Second] | X/Y | Z% | [Observations] |
| 3 | [Third] | X/Y | Z% | [Observations] |

## Head-to-Head (v20.0.2 variants only)

| Matchup | Result | Interpretation |
|---------|--------|----------------|
| Material vs Control | X-Y-Z (W-L-D) | [Analysis] |
| Positional vs Control | X-Y-Z (W-L-D) | [Analysis] |
| Material vs Positional | X-Y-Z (W-L-D) | [Analysis] |

## Tactical Accuracy

| Variant | Accuracy | Improvement |
|---------|----------|-------------|
| v20.0.2 Control | 10% | Baseline |
| Material | [X]% | [+Y%] |
| Positional | [X]% | [+Y%] |

## Decision

**Winner:** [Variant Name]
**Reason:** [Why this variant won]
**Next Steps:** [Deployment plan OR further testing needed]
```

---

## ✅ Checklist: Ready to Tournament

- [ ] `test_ab_variants.py` completes successfully
- [ ] All 3 BAT files added to Arena
- [ ] Tournament settings configured (6 engines, 20 games/pair, 5+3)
- [ ] Opening book disabled
- [ ] Adjudication rules set (50-move, 3-fold)
- [ ] Free time to run overnight (~4-6 hours for 300 games)
- [ ] Results template ready to fill
- [ ] Plan for analysis after completion

---

**Good luck with testing! 🎲**

The A/B test will reveal whether v7p3r's complex evaluation is helping or hurting the AI's move ordering effectiveness.
