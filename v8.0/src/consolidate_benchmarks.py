#!/usr/bin/env python3
"""
Consolidate Benchmark Reports and Update Catalog

Reads all benchmark report JSON files and updates the opponents_catalog.csv
with the latest/best benchmark results for each engine.

Usage:
    python consolidate_benchmarks.py
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BenchmarkConsolidator:
    """Consolidate benchmark reports and update catalog"""
    
    def __init__(self):
        self.benchmark_dir = Path(__file__).parent.parent / "benchmarks"
        self.catalog_path = Path(__file__).parent.parent / "docs" / "opponents_catalog.csv"
    
    def load_reports(self) -> Dict:
        """Load all benchmark reports from JSON files"""
        reports_by_engine = {}
        
        for report_file in self.benchmark_dir.glob("report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                engine_name = report.get('engine_name', 'Unknown')
                elo = report.get('estimated_elo', 0)
                
                # Keep best performing (highest ELO) report per engine
                if engine_name not in reports_by_engine:
                    reports_by_engine[engine_name] = {
                        'elo': elo,
                        'report': report,
                        'file': report_file.name
                    }
                elif elo > reports_by_engine[engine_name]['elo']:
                    print(f"  Updating {engine_name}: {reports_by_engine[engine_name]['elo']} -> {elo}")
                    reports_by_engine[engine_name] = {
                        'elo': elo,
                        'report': report,
                        'file': report_file.name
                    }
            except Exception as e:
                print(f"  ⚠️  Error reading {report_file.name}: {e}")
        
        return reports_by_engine
    
    def format_tier_summary(self, report: Dict) -> str:
        """Format tier performance as summary string"""
        tiers = report.get('tier_performance', [])
        tier_summaries = []
        
        for tier in tiers:
            tier_name = tier.get('tier_name', '').replace('tier', '').replace('_', ' ')
            solved = tier.get('solved', 0)
            total = tier.get('total_puzzles', 0)
            tier_summaries.append(f"{tier_name}: {solved}/{total}")
        
        return ", ".join(tier_summaries)
    
    def extract_relative_path(self, full_path: str) -> str:
        """Extract relative path after 'Tournament Engines/'"""
        # Full path: E:\...\Tournament Engines\V7P3R\V7P3R_v17.1\V7P3R_v17.1.bat
        # Want to match CSV paths like: V7P3R/V7P3R_v17.1/ or Cece/Cece_v1.0.exe
        
        path = full_path.replace('\\', '/')
        
        # Find "tournament engines/" and get everything after it
        if 'tournament engines/' in path.lower():
            idx = path.lower().rfind('tournament engines/')
            relative = path[idx + len('tournament engines/'):]
        else:
            relative = path
        
        return relative.lower()
    
    def match_paths(self, engine_path: str, csv_path: str) -> bool:
        """Check if engine path matches CSV path"""
        # Both paths after normalization
        engine_rel = self.extract_relative_path(engine_path)
        csv_rel = csv_path.lower().replace('\\', '/')
        
        # Normalize - remove trailing slashes and extensions for comparison
        engine_base = engine_rel.rsplit('/', 1)[0] if '/' in engine_rel else engine_rel
        engine_base = engine_base.rstrip('/')
        
        csv_base = csv_rel.rsplit('/', 1)[0] if '/' in csv_rel else csv_rel
        csv_base = csv_base.rstrip('/')
        
        # Direct match after normalization
        if engine_base == csv_base:
            return True
        
        # Component-based matching - extract version folder and engine folder
        engine_parts = engine_base.split('/')
        csv_parts = csv_base.split('/')
        
        # Match if last 1-2 path components match
        if len(engine_parts) > 0 and len(csv_parts) > 0:
            # Match last component (version folder like "c0br4_v3.1")
            if engine_parts[-1].lower() == csv_parts[-1].lower():
                return True
            
            # Match last two components (engine + version like "c0br4/c0br4_v3.1")
            if len(engine_parts) >= 2 and len(csv_parts) >= 2:
                if (engine_parts[-2].lower() == csv_parts[-2].lower() and 
                    engine_parts[-1].lower() == csv_parts[-1].lower()):
                    return True
        
        # Try folder name matching (e.g., "c0br4" matches "C0BR4")
        if len(engine_parts) > 0 and len(csv_parts) > 0:
            if engine_parts[-1].replace('_v', '_').split('_')[0] == csv_parts[-1].replace('_v', '_').split('_')[0]:
                return True
        
        return False
    
    def find_catalog_row_by_path(self, engine_path: str, csv_rows: List[Dict]) -> Optional[Dict]:
        """Find catalog row by matching engine path"""
        
        for row in csv_rows:
            csv_path = row.get('Path', '')
            if self.match_paths(engine_path, csv_path):
                return row
        
        return None
    
    def consolidate(self):
        """Consolidate reports and update catalog"""
        print("📊 Consolidating Benchmark Reports")
        print("="*60)
        
        # Load all reports
        print("📂 Loading benchmark reports...")
        reports_list = []
        for report_file in self.benchmark_dir.glob("report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    reports_list.append((report_file.name, report))
            except Exception as e:
                print(f"  ⚠️  Error reading {report_file.name}: {e}")
        
        print(f"✅ Loaded {len(reports_list)} benchmark reports\n")
        
        # Load current catalog
        print("📂 Loading catalog...")
        catalog_rows = []
        with open(self.catalog_path, 'r') as f:
            reader = csv.DictReader(f)
            catalog_rows = list(reader)
        print(f"✅ Loaded {len(catalog_rows)} engines from catalog\n")
        
        # Update catalog with benchmark data using path matching
        print("🔄 Updating catalog with benchmark results...")
        updated_count = 0
        skipped_count = 0
        
        for report_file, report in reports_list:
            engine_path = report.get('engine_path', '')
            engine_name = report.get('engine_name', 'Unknown')
            
            # Find matching row using path
            row = self.find_catalog_row_by_path(engine_path, catalog_rows)
            
            if row:
                elo = report.get('estimated_elo', 0)
                tier_summary = self.format_tier_summary(report)
                
                # Update row
                old_elo = row.get('Reference_ELO', '0')
                row['Reference_ELO'] = str(elo)
                row['Status'] = 'FUNCTIONAL' if elo >= 300 else 'BROKEN'
                row['Notes'] = f"Benchmark ELO {elo}, {tier_summary}"
                
                print(f"  ✅ {engine_name}: ELO {elo} (was {old_elo})")
                updated_count += 1
            else:
                print(f"  ⏭️  {engine_name}: No matching catalog entry for {engine_path}")
                skipped_count += 1
        
        print(f"\n✅ Updated {updated_count} engines")
        print(f"⏭️  Skipped {skipped_count} (no catalog match)\n")
        
        # Save updated catalog
        print("💾 Saving updated catalog...")
        with open(self.catalog_path, 'w', newline='') as f:
            if catalog_rows:
                fieldnames = list(catalog_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(catalog_rows)
        
        print(f"✅ Catalog saved: {self.catalog_path}\n")
        
        # Summary statistics
        print("="*60)
        print("📊 Summary Statistics")
        print("="*60)
        
        functional_count = sum(1 for r in catalog_rows if r.get('Status', '') == 'FUNCTIONAL')
        broken_count = sum(1 for r in catalog_rows if r.get('Status', '') == 'BROKEN')
        not_found_count = sum(1 for r in catalog_rows if r.get('Status', '') == 'NOT_FOUND')
        untested_count = sum(1 for r in catalog_rows if r.get('Status', '') == 'UNTESTED')
        
        print(f"Total engines: {len(catalog_rows)}")
        print(f"✅ Functional (tested): {functional_count}")
        print(f"❌ Broken: {broken_count}")
        print(f"🔲 Not found: {not_found_count}")
        print(f"❓ Untested: {untested_count}")
        
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
            print(f"\nELO Distribution (functional engines):")
            print(f"  Min: {min(elos)}")
            print(f"  Max: {max(elos)}")
            print(f"  Average: {sum(elos)/len(elos):.0f}")
            print(f"  Count: {len(elos)}")


def main():
    try:
        consolidator = BenchmarkConsolidator()
        consolidator.consolidate()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
