
# Arena Chess GUI Setup Instructions for V7P3R v2.0

## Step 1: Install the Engine

1. Open Arena Chess GUI
2. Go to **Engines** -> **Install New Engine**
3. Navigate to: `S:\Maker Stuff\Programming\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\tournament_packages\V7P3R_Tournament_v2.0_gen7_20250902_125314`
4. Select: `v7p3r_tournament_engine.py`
5. Engine Name: `V7P3R v2.0 Tournament`
6. Click **OK**

## Step 2: Configure Engine Settings

1. Go to **Engines** -> **Manage Engines**
2. Find "V7P3R v2.0 Tournament" in the list
3. Click **Details** and configure:
   - **Time per move**: 5-10 seconds (for testing)
   - **Depth**: Leave default (time-based)
   - **Threads**: 4 (adjust based on your CPU)
   - **Debug**: Off (unless troubleshooting)

## Step 3: Test the Engine

1. Go to **Engines** -> **New Engine Match**
2. Select V7P3R v2.0 as Player 1
3. Select another engine as Player 2
4. Set time control: 5+3 (5 minutes + 3 second increment)
5. Start the match

## Step 4: Tournament Setup

### Quick Tournament:
1. **Engines** -> **Tournament**
2. Add V7P3R v2.0 and other engines
3. Set time control: 15+5 or 10+3
4. Set rounds: 10-20 games per opponent
5. Start tournament

### Swiss Tournament:
1. Create a Swiss system tournament
2. Time control: 15+10 (standard tournament time)
3. Add 6-8 engines of similar strength
4. Set 7-9 rounds
5. Monitor results

## Recommended Opponents for Rating

**Beginner Level (800-1200):**
- Fairy-Max
- Micro-Max  
- Vice

**Intermediate Level (1200-1800):**
- Fruit 2.1
- Crafty
- GNU Chess

**Advanced Level (1800-2200):**
- Stockfish (limited depth/time)
- Komodo (limited)
- Houdini (limited)

## Tournament Recording

Record results in this format:

```
Date: 2025-09-02
Tournament: [Tournament Name]
Time Control: [e.g., 15+5]
Opponents: [List engines]
Score: X/Y ([Win-Draw-Loss])
Performance Rating: [Estimate]
Notes: [Key observations]
```

## Troubleshooting

If the engine doesn't start:
1. Check Python installation
2. Verify PyTorch installation: `pip install torch`
3. Run `python v7p3r_tournament_engine.py` directly
4. Check the Arena error log

## Performance Expectations

Based on training data:
- **Estimated Rating**: 1400-1800 ELO
- **Style**: Tactical, aggressive
- **Strengths**: Piece coordination, tactical shots
- **Weaknesses**: May be inconsistent in pure endgames

Good luck in your tournaments! 🏆
