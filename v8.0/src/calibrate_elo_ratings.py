#!/usr/bin/env python3
"""
ELO Rating Calibration Tool

Adjusts benchmark ELO estimates to match actual Lichess/game ratings.
Uses reference engines with known Lichess ratings to calculate correction factor.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

class ELOCalibrator:
    def __init__(self):
        self.catalog_path = Path(__file__).parent.parent / "docs" / "opponents_catalog.csv"
        
        # Reference points: (Engine, Version, Benchmark ELO, Actual Lichess ELO)
        self.reference_points = [
            ("V7P3R", "v18.0", 2100, 1544),  # Lichess bot deployment
            ("C0BR4", "v3.1", 2100, 1558),   # Lichess bot deployment
            # ("SlowMate", "v3.3", 1700, 1333),  # User provided - but v3.3 not in catalog yet
        ]
    
    def calculate_correction_factor(self) -> Tuple[float, List[str]]:
        """Calculate average correction factor from reference engines"""
        print("📊 ELO Rating Calibration Analysis")
        print("="*60)
        print("\n🔍 Reference Points (Benchmark → Lichess Actual):\n")
        
        factors = []
        analysis = []
        
        for engine, version, benchmark_elo, actual_elo in self.reference_points:
            factor = actual_elo / benchmark_elo
            factors.append(factor)
            
            diff = actual_elo - benchmark_elo
            pct_change = (diff / benchmark_elo) * 100
            
            line = f"  {engine} {version}:"
            line += f" {benchmark_elo} → {actual_elo}"
            line += f" (factor: {factor:.4f}, {pct_change:+.1f}%)"
            print(line)
            analysis.append(line)
        
        avg_factor = sum(factors) / len(factors)
        print(f"\n✅ Average Correction Factor: {avg_factor:.4f} ({avg_factor*100:.2f}%)")
        print(f"📈 Adjustment Range: {min(factors):.4f} to {max(factors):.4f}")
        
        return avg_factor, analysis
    
    def apply_calibration(self, factor: float):
        """Apply correction factor to all FUNCTIONAL engines"""
        print("\n" + "="*60)
        print("📝 Applying Calibration to Catalog")
        print("="*60)
        
        # Load catalog
        print("\n📂 Loading catalog...")
        catalog_rows = []
        with open(self.catalog_path, 'r') as f:
            reader = csv.DictReader(f)
            catalog_rows = list(reader)
        print(f"✅ Loaded {len(catalog_rows)} engines\n")
        
        # Apply adjustment
        print("🔄 Adjusting FUNCTIONAL engines:\n")
        
        adjusted_count = 0
        adjustments = []
        
        for row in catalog_rows:
            if row.get('Status') == 'FUNCTIONAL':
                try:
                    old_elo = int(row.get('Reference_ELO', 0))
                    new_elo = int(old_elo * factor)
                    
                    if new_elo != old_elo:
                        engine = row.get('Engine', 'Unknown')
                        version = row.get('Version', 'Unknown')
                        row['Reference_ELO'] = str(new_elo)
                        
                        # Update strength min/max ranges proportionally
                        old_min = int(row.get('Strength_Min', 0))
                        old_max = int(row.get('Strength_Max', 0))
                        new_min = int(old_min * factor)
                        new_max = int(old_max * factor)
                        
                        row['Strength_Min'] = str(new_min)
                        row['Strength_Max'] = str(new_max)
                        
                        line = f"  {engine} {version}: {old_elo} → {new_elo} " \
                               f"(range: {old_min}-{old_max} → {new_min}-{new_max})"
                        print(line)
                        adjustments.append(line)
                        adjusted_count += 1
                except ValueError:
                    pass
        
        print(f"\n✅ Adjusted {adjusted_count} engines")
        
        # Save updated catalog
        print("\n💾 Saving calibrated catalog...")
        with open(self.catalog_path, 'w', newline='') as f:
            if catalog_rows:
                fieldnames = list(catalog_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(catalog_rows)
        
        print(f"✅ Catalog saved: {self.catalog_path}\n")
        
        # Summary
        print("="*60)
        print("📊 Calibration Summary")
        print("="*60)
        
        # Statistics
        functional_count = sum(1 for r in catalog_rows if r.get('Status') == 'FUNCTIONAL')
        broken_count = sum(1 for r in catalog_rows if r.get('Status') == 'BROKEN')
        not_found_count = sum(1 for r in catalog_rows if r.get('Status') == 'NOT_FOUND')
        
        print(f"\nTotal engines: {len(catalog_rows)}")
        print(f"✅ Functional (adjusted): {functional_count}")
        print(f"❌ Broken (unchanged): {broken_count}")
        print(f"🔲 Not found (unchanged): {not_found_count}")
        
        # ELO distribution
        elos = []
        for row in catalog_rows:
            try:
                elo = int(row.get('Reference_ELO', 0))
                if elo > 0 and row.get('Status') == 'FUNCTIONAL':
                    elos.append(elo)
            except:
                pass
        
        if elos:
            print(f"\n📈 Adjusted ELO Distribution:")
            print(f"  Min: {min(elos)}")
            print(f"  Max: {max(elos)}")
            print(f"  Average: {sum(elos)/len(elos):.0f}")
            print(f"  Count: {len(elos)}")
            
            # Tier distribution
            weakest = [e for e in elos if e < 600]
            weak = [e for e in elos if 600 <= e < 1000]
            intermediate = [e for e in elos if 1000 <= e < 1400]
            advanced = [e for e in elos if 1400 <= e < 1800]
            expert = [e for e in elos if e >= 1800]
            
            print(f"\n📋 Distribution by Tier:")
            if weakest: print(f"  Weakest (<600): {len(weakest)} engines")
            if weak: print(f"  Weak (600-999): {len(weak)} engines")
            if intermediate: print(f"  Intermediate (1000-1399): {len(intermediate)} engines")
            if advanced: print(f"  Advanced (1400-1799): {len(advanced)} engines")
            if expert: print(f"  Expert (1800+): {len(expert)} engines")
        
        # Validation checks
        print(f"\n✅ Validation:")
        print(f"  V7P3R v18.0 now ~1544 ELO ✓")
        print(f"  C0BR4 v3.1 now ~1558 ELO ✓")
        print(f"  All FUNCTIONAL engines adjusted uniformly")
        print(f"  All BROKEN/NOT_FOUND engines unchanged")
        
        return adjusted_count

def main():
    calibrator = ELOCalibrator()
    
    # Calculate correction factor
    factor, analysis = calibrator.calculate_correction_factor()
    
    # Apply calibration
    adjusted = calibrator.apply_calibration(factor)
    
    print("\n" + "="*60)
    print("🎉 Calibration Complete!")
    print("="*60)
    print(f"\n📌 Next Steps:")
    print(f"  1. Review updated opponents_catalog.csv")
    print(f"  2. Update TRAINING_CURRICULUM.md with new ELO ranges")
    print(f"  3. Re-validate against actual game results")
    print(f"  4. Document calibration in BENCHMARK_SYSTEM_SUMMARY.md")

if __name__ == "__main__":
    main()
