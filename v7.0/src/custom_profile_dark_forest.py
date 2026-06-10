"""
V7P3R Custom Personality Profile - "Dark Forest Assassin"

User preferences (June 2026):
1. Ultra-aggressive (Tal/Shirov) - sacrifice freely for initiative
2. Selective king exposure - only when attacking opponent
3. Ultra-sharp chaos - maximum tactical complexity
4. Free material sacrifice - any compensation justifies it
5. Classical/Hypermodern hybrid - flexible center approach
6. Technical precision + Fighting spirit - convert advantages, never accept draws
7. Practical time management - simplify when needed

Result: Chaotic middlegame tactician → Technical endgame converter
"""

import json
from personality_rewards import PersonalityWeights
from personality_tuner import PlaystyleProfile


# Dark Forest Assassin Weights
dark_forest_weights = PersonalityWeights(
    # ULTRA-AGGRESSIVE COMPLEXITY (Questions 1, 3, 4)
    forest_darkness=0.20,          # Maximum complexity seeking (vs 0.15 Tal default)
    piece_tension=0.15,            # Ultra-sharp positions (vs 0.10 default)
    move_diversity=0.08,           # Encourage all pieces active
    
    # FREE MATERIAL SACRIFICE (Question 4)
    material_sacrifice_bonus=0.15, # Highest bonus for sacrifices (vs 0.10 default)
    material_threshold=7,          # Tolerate up to 7 pawns loss (vs 5 default)
    complexity_threshold=0.15,     # Lower threshold (easier to justify sacrifice)
    
    # SELECTIVE KING AGGRESSION (Question 2)
    king_risk_penalty=-0.03,       # Very tolerant of king exposure (vs -0.05)
    king_risk_tolerance=3.0,       # Accept up to 3 pawn shield loss (vs 2.0)
    attack_bonus=0.12,             # Huge bonus for attacking opponent king (vs 0.08)
    
    # CLASSICAL/HYPERMODERN HYBRID (Question 5)
    center_control=0.07,           # Strong center emphasis (between classical/hypermodern)
    active_rooks=0.05,             # Rook activity matters
    
    # TECHNICAL ENDGAME CONVERSION (Question 6 - Technical precision)
    passed_pawns=0.05,             # Strong endgame fundamentals
    bishop_pair=0.03,              # Long-term advantages
    endgame_complexity_weight=0.4, # Still seeks some complexity in endgames
    
    # Note: Fighting chess aspect handled by draw threshold (never repeat < +0.50)
)


# Create profile
dark_forest_profile = PlaystyleProfile(
    name="DarkForestAssassin",
    description=(
        "Ultra-aggressive middlegame tactician with technical endgame conversion. "
        "Freely sacrifices material for initiative and chaos. "
        "Selectively exposes king when attacking. "
        "Never accepts draws in fighting positions. "
        "Transitions from sharp tactics to technical precision."
    ),
    weights=dark_forest_weights,
    example_positions=[
        # Add example positions after testing
    ],
    author="V7P3R User",
    version="v7.0"
)


# Save profile
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Create profiles directory
    profiles_dir = Path("../../profiles")
    profiles_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    profile_path = profiles_dir / "dark_forest_assassin.json"
    with open(profile_path, 'w') as f:
        json.dump(dark_forest_profile.to_dict(), f, indent=2)
    
    print("="*60)
    print("DARK FOREST ASSASSIN - V7P3R Custom Personality")
    print("="*60)
    
    print("\n🎯 Profile Created Successfully!")
    print(f"  Name: {dark_forest_profile.name}")
    print(f"  Description: {dark_forest_profile.description}")
    print(f"  Saved to: {profile_path}")
    
    print("\n⚔️ Key Characteristics:")
    print(f"  Forest Darkness Weight: {dark_forest_weights.forest_darkness} (vs 0.15 Tal)")
    print(f"  Piece Tension Weight: {dark_forest_weights.piece_tension} (vs 0.10 Tal)")
    print(f"  Material Sacrifice Bonus: {dark_forest_weights.material_sacrifice_bonus}")
    print(f"  Material Threshold: {dark_forest_weights.material_threshold} pawns")
    print(f"  Attack Bonus: {dark_forest_weights.attack_bonus} (vs 0.08 Tal)")
    print(f"  King Risk Penalty: {dark_forest_weights.king_risk_penalty} (vs -0.05 Tal)")
    
    print("\n🔥 Personality Comparison:")
    print(f"  More aggressive than Tal: ✓")
    print(f"  Higher complexity seeking: ✓")
    print(f"  More material sacrifice tolerance: ✓")
    print(f"  Better endgame technique: ✓")
    print(f"  Fighting spirit (no easy draws): ✓")
    
    print("\n📊 Weight Distribution:")
    print(f"  Complexity rewards: {dark_forest_weights.forest_darkness + dark_forest_weights.piece_tension + dark_forest_weights.move_diversity:.2f}")
    print(f"  Material sacrifice: {dark_forest_weights.material_sacrifice_bonus:.2f}")
    print(f"  Attack/King: {dark_forest_weights.attack_bonus + abs(dark_forest_weights.king_risk_penalty):.2f}")
    print(f"  Strategic: {dark_forest_weights.center_control + dark_forest_weights.passed_pawns + dark_forest_weights.bishop_pair + dark_forest_weights.active_rooks:.2f}")
    print(f"  Total personality weight: ~0.86 (high)")
    
    print("\n🎮 Recommended Training Mix:")
    print(f"  Stockfish evaluation: 70%")
    print(f"  Personality rewards: 20%")
    print(f"  Game outcome: 10%")
    
    print("\n🚫 Anti-Patterns (What V7P3R Should Avoid):")
    print(f"  ❌ Passive, defensive positions (low forest_darkness)")
    print(f"  ❌ Material-first thinking (will sacrifice freely)")
    print(f"  ❌ Accepting draws in equal positions (fighting chess)")
    print(f"  ❌ King safety paranoia (selective risk tolerance)")
    print(f"  ❌ Simplifying when winning (prefers maintaining pressure)")
    
    print("\n✅ Desired Patterns (What V7P3R Should Seek):")
    print(f"  ✅ Sharp tactical positions (forest_darkness > 0.4)")
    print(f"  ✅ Multiple piece attacks on opponent king")
    print(f"  ✅ Material sacrifices with compensation (initiative, attack, development)")
    print(f"  ✅ Center control (classical e4/d4 or hypermodern pressure)")
    print(f"  ✅ Passed pawns in endgames (technical conversion)")
    print(f"  ✅ Active rook placement (7th rank, open files)")
    
    print("\n🎯 Expected Playing Style:")
    print(f"  Opening: Classical center occupation OR hypermodern control")
    print(f"          Rapid development, aggressive piece placement")
    print(f"  Middlegame: Maximum tactical chaos, free sacrifices")
    print(f"             Attack opponent king relentlessly")
    print(f"             Tolerate king exposure if attacking")
    print(f"  Endgame: Technical precision, convert advantages")
    print(f"          Create passed pawns, activate king")
    print(f"          Never accept draws prematurely")
    
    print("\n📈 Next Steps:")
    print(f"  1. Test profile on tactical position suite")
    print(f"  2. Verify personality emerges in self-play")
    print(f"  3. Compare vs Tal profile (should be more aggressive)")
    print(f"  4. Fine-tune weights based on gameplay observation")
    print(f"  5. Train V7 network with these reward weights")
    
    print(f"\n🔧 Tuning Commands:")
    print(f"  # Load profile")
    print(f"  from personality_tuner import PersonalityTuner")
    print(f"  tuner = PersonalityTuner()")
    print(f"  profile = tuner.load_profile('{profile_path}')")
    print(f"  ")
    print(f"  # Test on position")
    print(f"  calculator = PersonalityRewardCalculator(profile.weights)")
    print(f"  reward = calculator.calculate_total_reward(features_dict, stockfish_eval=0.2)")
    print(f"  ")
    print(f"  # Compare to Tal")
    print(f"  comparison = tuner.compare_profiles('darkforestassassin', 'tal')")
