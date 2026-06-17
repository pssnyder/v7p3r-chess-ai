# DatasetAnalyzer: Analyze Parquet files in chunks, compute statistics, create splits
import os
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# NULL value constants
NULL_EVAL = 32767
NULL_DEPTH = 255
NULL_TIME = 4294967295
NULL_CLOCK = 65535
NULL_WDL = 127

class DatasetAnalyzer:
    """Analyze Parquet files in chunks, compute statistics, create splits"""
    
    def __init__(self, parquet_dir: str, chunk_size: int = 100_000):
        self.parquet_dir = parquet_dir
        self.chunk_size = chunk_size
        self.metadata = {}
        self.validation_results = None  # Store validation results
        self.duplicate_stats = None  # Store duplicate FEN statistics
    
    def compute_statistics(self) -> Dict:
        """Read most recent Parquet file, compute statistics for each feature"""
        print("Computing dataset statistics...")
        
        parquet_dir = Path(self.parquet_dir)
        parquet_files = list(parquet_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No Parquet files found in {self.parquet_dir}")
            return {}
        
        # Get most recent file by modification time
        most_recent = max(parquet_files, key=lambda f: f.stat().st_mtime)
        parquet_files = [most_recent]
        print(f"Processing most recent file: {most_recent.name}")
        
        stats = {
            'evaluation': {'values': [], 'histogram': defaultdict(int)},
            'depth': {'values': [], 'histogram': defaultdict(int)},
            'time': {'values': [], 'histogram': defaultdict(int)},
            'clock': {'values': [], 'histogram': defaultdict(int)},
            'wdl': {'distribution': defaultdict(int)},
            'material': {'values': [], 'histogram': defaultdict(int)},
            'phase': {'distribution': defaultdict(int)},
            'piece_count': {'distribution': defaultdict(int)},
            'total_positions': 0
        }
        
        # Process each Parquet file
        for parquet_file in parquet_files:
            print(f"  Processing {parquet_file.name}...")
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            # Skip null values (32767 for eval, 255 for depth, etc.)
            eval_mask = df['evaluation'] != 32767
            depth_mask = df['depth'] != 255
            time_mask = df['time'] != 4294967295
            clock_mask = df['clock'] != 65535
            
            # Collect statistics
            stats['evaluation']['values'].extend(df[eval_mask]['evaluation'].values)
            stats['depth']['values'].extend(df[depth_mask]['depth'].values)
            stats['time']['values'].extend(df[time_mask]['time'].values)
            stats['clock']['values'].extend(df[clock_mask]['clock'].values)
            stats['material']['values'].extend(df['material'].values)
            
            # WDL distribution
            for wdl_val in df['wdl'].values:
                if wdl_val in [-1, 0, 1]:
                    stats['wdl']['distribution'][str(wdl_val)] += 1
            
            # Phase distribution
            for phase_val in df['phase'].values:
                stats['phase']['distribution'][str(int(phase_val))] += 1
            
            # Piece count distribution
            for pc_val in df['piece_count'].values:
                stats['piece_count']['distribution'][str(int(pc_val))] += 1
            
            stats['total_positions'] += len(df)
        
        # Compute aggregates
        self.stats = {
            'total_positions': stats['total_positions'],
            'evaluation': {
                'mean': float(np.mean(stats['evaluation']['values'])),
                'std': float(np.std(stats['evaluation']['values'])),
                'min': float(np.min(stats['evaluation']['values'])),
                'max': float(np.max(stats['evaluation']['values'])),
                'median': float(np.median(stats['evaluation']['values'])),
                'q25': float(np.percentile(stats['evaluation']['values'], 25)),
                'q75': float(np.percentile(stats['evaluation']['values'], 75)),
            },
            'depth': {
                'mean': float(np.mean(stats['depth']['values'])),
                'std': float(np.std(stats['depth']['values'])),
                'min': float(np.min(stats['depth']['values'])),
                'max': float(np.max(stats['depth']['values'])),
            },
            'wdl_distribution': dict(stats['wdl']['distribution']),
            'phase_distribution': dict(stats['phase']['distribution']),
            'material': {
                'mean': float(np.mean(stats['material']['values'])),
                'min': float(np.min(stats['material']['values'])),
                'max': float(np.max(stats['material']['values'])),
            },
            'piece_count_distribution': dict(stats['piece_count']['distribution']),
        }
        
        print(f"✓ Computed statistics for {self.stats['total_positions']} positions")
        return self.stats
    
    def create_train_val_test_split(self, train_pct: float = 0.7, val_pct: float = 0.15, test_pct: float = 0.15) -> Dict:
        """Create deterministic train/val/test splits based on fen_hash"""
        print(f"Creating train/val/test splits ({train_pct}/{val_pct}/{test_pct})...")
        
        parquet_dir = Path(self.parquet_dir)
        parquet_files = list(parquet_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No Parquet files found in {self.parquet_dir}")
            return {}
        
        # Get most recent file by modification time
        most_recent = max(parquet_files, key=lambda f: f.stat().st_mtime)
        parquet_files = [most_recent]
        print(f"Processing most recent file: {most_recent.name}")
        
        splits = {
            'train': [],
            'val': [],
            'test': [],
            'total_samples': 0,
            'split_info': {
                'train_pct': train_pct,
                'val_pct': val_pct,
                'test_pct': test_pct,
                'method': 'fen_hash modulo'
            }
        }
        
        row_offset = 0
        
        for parquet_file in parquet_files:
            print(f"  Processing {parquet_file.name} for splits...")
            table = pq.read_table(parquet_file, columns=['fen_hash'])
            fen_hashes = table['fen_hash'].to_pylist()
            
            for i, fen_hash in enumerate(fen_hashes):
                global_row_idx = row_offset + i
                # Use fen_hash modulo for reproducible, random-like assignment
                hash_mod = fen_hash % 100
                
                if hash_mod < train_pct * 100:
                    splits['train'].append(global_row_idx)
                elif hash_mod < (train_pct + val_pct) * 100:
                    splits['val'].append(global_row_idx)
                else:
                    splits['test'].append(global_row_idx)
            
            row_offset += len(fen_hashes)
            splits['total_samples'] = row_offset
        
        # Print summary
        print(f"✓ Split Summary:")
        print(f"  Train: {len(splits['train'])} ({len(splits['train'])/splits['total_samples']*100:.1f}%)")
        print(f"  Val:   {len(splits['val'])} ({len(splits['val'])/splits['total_samples']*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/splits['total_samples']*100:.1f}%)")
        
        return splits
    
    def generate_metadata_report(self, output_dir: str = "v10.0/analysis") -> str:
        """Generate HTML/JSON report with statistics and visualizations"""
        print("Generating metadata report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute duplicate FEN statistics
        duplicate_stats = self.check_duplicate_fens()
        self.duplicate_stats = duplicate_stats
        
        # Save JSON statistics (include duplicate stats)
        json_file = output_path / "dataset_stats.json"
        combined_stats = {**self.stats, 'duplicate_fens': duplicate_stats}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2)
        print(f"✓ Saved JSON stats: {json_file}")
        
        # Generate HTML report
        html_content = self._generate_html_report()
        html_file = output_path / "dataset_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Saved HTML report: {html_file}")
        
        return str(html_file)
    
    def check_duplicate_fens(self, parquet_file: Path = None, show_top_n: int = 20) -> Dict:
        """
        Check for duplicate FEN hashes in the dataset.
        Returns statistics on uniqueness and top duplicated positions.
        """
        if parquet_file is None:
            parquet_dir = Path(self.parquet_dir)
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if not parquet_files:
                return {}
            parquet_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
        
        print(f"\nChecking for duplicate FEN hashes...")
        table = pq.read_table(parquet_file, columns=['fen_hash'])
        fen_hashes = table['fen_hash'].to_pylist()
        
        total_hashes = len(fen_hashes)
        unique_hashes = len(set(fen_hashes))
        duplicate_count = total_hashes - unique_hashes
        duplicate_pct = (duplicate_count / total_hashes * 100) if total_hashes > 0 else 0
        
        # Count how many times each hash appears
        hash_counts = {}
        for h in fen_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        
        # Find most duplicated hashes
        top_duplicates = sorted(
            [(h, count) for h, count in hash_counts.items() if count > 1],
            key=lambda x: x[1],
            reverse=True
        )[:show_top_n]
        
        duplicate_summary = {
            'total_positions': total_hashes,
            'unique_fen_hashes': unique_hashes,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_pct,
            'top_duplicated_hashes': [
                {'fen_hash': h, 'occurrence_count': count}
                for h, count in top_duplicates
            ]
        }
        
        print(f"✓ FEN Hash Duplicate Check Complete")
        print(f"  Total positions:      {total_hashes:,}")
        print(f"  Unique FEN hashes:    {unique_hashes:,}")
        print(f"  Duplicate positions:  {duplicate_count:,} ({duplicate_pct:.2f}%)")
        if top_duplicates:
            print(f"\n  Top {len(top_duplicates)} most duplicated hashes:")
            for h, count in top_duplicates[:5]:
                print(f"    Hash {h}: appears {count} times")
        
        return duplicate_summary
    
    def _generate_html_report(self) -> str:
        """Generate HTML with embedded statistics"""
        if not self.stats:
            return "<h1>No statistics computed yet. Run compute_statistics() first.</h1>"
        
        wdl_data = self.stats.get('wdl_distribution', {})
        phase_data = self.stats.get('phase_distribution', {})
        
        # Extract WDL values correctly (keys are strings: '1', '0', '-1')
        wdl_wins = wdl_data.get('1', 0)
        wdl_draws = wdl_data.get('0', 0)
        wdl_losses = wdl_data.get('-1', 0)
        
        phase_labels = sorted([str(i) for i in range(25)])
        phase_values = [self.stats['phase_distribution'].get(str(i), 0) for i in range(25)]
        
        # Build validation section HTML
        validation_html = ""
        if self.validation_results:
            null_vals = self.validation_results.get('null_values', {})
            consistency = self.validation_results.get('eval_wdl_consistency', {})
            derivable = self.validation_results.get('fen_derivable_fields', {})
            
            validation_html = f"""
        <h2>Data Quality Validation</h2>
        
        <h3>NULL Value Detection</h3>
        <table>
            <tr>
                <th>Field</th>
                <th>NULL Count</th>
                <th>Percentage</th>
                <th>Status</th>
            </tr>
"""
            if null_vals.get('null_counts'):
                for field, count in null_vals['null_counts'].items():
                    pct = null_vals['null_percentages'].get(field, 0)
                    status = "✓ OK" if count == 0 else "⚠ Found NULLs"
                    validation_html += f"""            <tr>
                <td>{field}</td>
                <td>{count:,}</td>
                <td>{pct:.2f}%</td>
                <td>{status}</td>
            </tr>
"""
            validation_html += """        </table>
        
        <h3>Eval → WDL Consistency</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
            if consistency:
                total_evals = consistency.get('eval_rows_total', 0)
                violations = consistency.get('violations', 0)
                viol_pct = consistency.get('violation_percentage', 0)
                validation_html += f"""            <tr>
                <td>Total rows with evaluation</td>
                <td>{total_evals:,}</td>
            </tr>
            <tr>
                <td>Eval↔WDL inconsistencies</td>
                <td>{violations:,} ({viol_pct:.2f}%)</td>
            </tr>
            <tr>
                <td>Status</td>
                <td>{'✓ PASS' if violations == 0 else '⚠ VIOLATIONS FOUND'}</td>
            </tr>
"""
            validation_html += """        </table>
        
        <h3>FEN-Derivable Fields</h3>
        <table>
            <tr>
                <th>Field</th>
                <th>NULL Count</th>
                <th>Percentage</th>
                <th>Status</th>
            </tr>
"""
            if derivable:
                for field in ['piece_count', 'material', 'phase']:
                    info = derivable.get(field, {})
                    null_count = info.get('null_count', 0)
                    null_pct = info.get('null_percentage', 0)
                    status = info.get('status', '?')
                    validation_html += f"""            <tr>
                <td>{field}</td>
                <td>{null_count:,}</td>
                <td>{null_pct:.2f}%</td>
                <td>{status} {'OK' if null_count == 0 else 'ERRORS'}</td>
            </tr>
"""
            validation_html += """        </table>
"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .stat-box {{ 
            background: #f0f0f0; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .chart-container {{ 
            position: relative; 
            width: 100%; 
            height: 400px;
            margin: 30px 0;
        }}
        h1 {{ color: #2196F3; }}
        h2 {{ color: #555; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
        h3 {{ color: #666; margin-top: 25px; }}
        table {{ 
            border-collapse: collapse; 
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left;
        }}
        th {{ background-color: #2196F3; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dataset Analysis Report</h1>
        
        {validation_html}
        
        <h2>Dataset Overview</h2>
        <div class="stat-box">
            <strong>Total Positions:</strong> {self.stats['total_positions']:,}
        </div>
        
        <h2>Duplicate FEN Hash Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>"""
        
        # Add duplicate FEN stats if available
        if hasattr(self, 'duplicate_stats') and self.duplicate_stats:
            dup_total = self.duplicate_stats.get('total_positions', 0)
            dup_unique = self.duplicate_stats.get('unique_fen_hashes', 0)
            dup_count = self.duplicate_stats.get('duplicate_count', 0)
            dup_pct = self.duplicate_stats.get('duplicate_percentage', 0)
            
            html += f"""            <tr>
                <td>Total Positions</td>
                <td>{dup_total:,}</td>
            </tr>
            <tr>
                <td>Unique FEN Hashes</td>
                <td>{dup_unique:,}</td>
            </tr>
            <tr>
                <td>Duplicate Positions</td>
                <td>{dup_count:,} ({dup_pct:.2f}%)</td>
            </tr>
            <tr>
                <td>Uniqueness Ratio</td>
                <td>{(dup_unique/dup_total*100):.2f}% unique</td>
            </tr>
        </table>
"""
            # Add top duplicates table if available
            top_dups = self.duplicate_stats.get('top_duplicated_hashes', [])
            if top_dups:
                html += """        <h3>Top Duplicated FEN Hashes</h3>
        <table>
            <tr>
                <th>FEN Hash</th>
                <th>Occurrence Count</th>
            </tr>
"""
                for dup_entry in top_dups[:10]:  # Show top 10
                    fen_hash = dup_entry.get('fen_hash', 'N/A')
                    count = dup_entry.get('occurrence_count', 0)
                    html += f"""            <tr>
                <td>{fen_hash}</td>
                <td>{count}</td>
            </tr>
"""
                html += """        </table>
"""
        else:
            html += """            <tr>
                <td>Duplicate Analysis</td>
                <td>⚠ Not computed</td>
            </tr>
        </table>
"""
        
        html += f"""
        <h2>Evaluation Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{self.stats['evaluation']['mean']:.2f}</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>{self.stats['evaluation']['std']:.2f}</td>
            </tr>
            <tr>
                <td>Min</td>
                <td>{self.stats['evaluation']['min']:.0f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{self.stats['evaluation']['max']:.0f}</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{self.stats['evaluation']['median']:.2f}</td>
            </tr>
            <tr>
                <td>Q25</td>
                <td>{self.stats['evaluation']['q25']:.2f}</td>
            </tr>
            <tr>
                <td>Q75</td>
                <td>{self.stats['evaluation']['q75']:.2f}</td>
            </tr>
        </table>
        
        <h2>Depth Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{self.stats['depth']['mean']:.2f}</td>
            </tr>
            <tr>
                <td>Min</td>
                <td>{self.stats['depth']['min']:.0f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{self.stats['depth']['max']:.0f}</td>
            </tr>
        </table>
        
        <h2>Win/Draw/Loss Distribution</h2>
        <div class="chart-container">
            <canvas id="wdlChart"></canvas>
        </div>
        
        <h2>Game Phase Distribution</h2>
        <div class="chart-container">
            <canvas id="phaseChart"></canvas>
        </div>
        
        <h2>Material Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{self.stats['material']['mean']:.0f}</td>
            </tr>
            <tr>
                <td>Min</td>
                <td>{self.stats['material']['min']:.0f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{self.stats['material']['max']:.0f}</td>
            </tr>
        </table>
    </div>
    
    <script>
        // WDL Chart - Fixed to show correct values
        const wdlCtx = document.getElementById('wdlChart').getContext('2d');
        new Chart(wdlCtx, {{
            type: 'pie',
            data: {{
                labels: ['Wins (wdl=1)', 'Draws (wdl=0)', 'Losses (wdl=-1)'],
                datasets: [{{
                    data: [{wdl_wins}, {wdl_draws}, {wdl_losses}],
                    backgroundColor: ['#4CAF50', '#FFC107', '#F44336']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
        
        // Phase Chart
        const phaseCtx = document.getElementById('phaseChart').getContext('2d');
        new Chart(phaseCtx, {{
            type: 'bar',
            data: {{
                labels: {phase_labels},
                datasets: [{{
                    label: 'Count',
                    data: {phase_values},
                    backgroundColor: '#2196F3'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html


# ============================================================================
# VALIDATION METHODS
# ============================================================================

    def validate_null_values(self, parquet_file: Path = None) -> Dict:
        """
        Check for NULL placeholder values in dataset.
        Returns count of nulls for each field.
        """
        if parquet_file is None:
            parquet_dir = Path(self.parquet_dir)
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if not parquet_files:
                return {}
            parquet_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
        
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        null_counts = {
            'evaluation': (df['evaluation'] == NULL_EVAL).sum(),
            'depth': (df['depth'] == NULL_DEPTH).sum(),
            'time': (df['time'] == NULL_TIME).sum(),
            'clock': (df['clock'] == NULL_CLOCK).sum(),
            'wdl': (df['wdl'] == NULL_WDL).sum(),
        }
        
        total_rows = len(df)
        null_summary = {
            'total_rows': total_rows,
            'null_counts': null_counts,
            'null_percentages': {
                k: (v / total_rows * 100) if total_rows > 0 else 0
                for k, v in null_counts.items()
            }
        }
        
        return null_summary
    
    def validate_eval_wdl_consistency(self, parquet_file: Path = None) -> Dict:
        """
        Check if evaluation values are consistent with WDL labels.
        Rules:
        - eval > 200cp → wdl should be 1 (winning)
        - eval < -200cp → wdl should be -1 (losing)
        - -200 <= eval <= 200cp → wdl should be 0 (drawn)
        """
        if parquet_file is None:
            parquet_dir = Path(self.parquet_dir)
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if not parquet_files:
                return {}
            parquet_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
        
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Only check rows where evaluation is NOT NULL
        eval_mask = df['evaluation'] != NULL_EVAL
        eval_present = df[eval_mask].copy()
        
        violations = []
        
        for idx, row in eval_present.iterrows():
            eval_cp = row['evaluation']
            wdl_actual = row['wdl']
            
            # Skip if WDL is NULL
            if wdl_actual == NULL_WDL:
                continue
            
            # Determine expected WDL from evaluation
            if eval_cp > 200:
                wdl_expected = 1
            elif eval_cp < -200:
                wdl_expected = -1
            else:
                wdl_expected = 0
            
            # Check for violation
            if wdl_actual != wdl_expected:
                violations.append({
                    'index': idx,
                    'eval_cp': eval_cp,
                    'wdl_expected': wdl_expected,
                    'wdl_actual': wdl_actual,
                })
        
        violation_count = len(violations)
        eval_rows_with_valid_wdl = len(eval_present[eval_present['wdl'] != NULL_WDL])
        
        consistency_summary = {
            'eval_rows_total': len(eval_present),
            'eval_rows_with_valid_wdl': eval_rows_with_valid_wdl,
            'violations': violation_count,
            'violation_percentage': (violation_count / eval_rows_with_valid_wdl * 100) if eval_rows_with_valid_wdl > 0 else 0,
            'violation_details': violations[:20]  # Return top 20 for inspection
        }
        
        return consistency_summary
    
    def validate_fen_derivable_fields(self, parquet_file: Path = None) -> Dict:
        """
        Validate that FEN-derivable fields are never NULL.
        piece_count, material, and phase can always be calculated from FEN.
        """
        if parquet_file is None:
            parquet_dir = Path(self.parquet_dir)
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if not parquet_files:
                return {}
            parquet_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
        
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Check for NULL values in FEN-derivable fields
        piece_count_nulls = (df['piece_count'] == 255).sum()
        material_nulls = (df['material'] == 65535).sum()
        phase_nulls = (df['phase'] == 255).sum()
        
        # Also check for zeros (which could indicate missing/empty positions)
        piece_count_zeros = (df['piece_count'] == 0).sum()
        material_zeros = (df['material'] == 0).sum()
        phase_zeros = (df['phase'] == 0).sum()
        
        total_rows = len(df)
        
        derivable_summary = {
            'total_rows': total_rows,
            'piece_count': {
                'null_count': piece_count_nulls,
                'null_percentage': (piece_count_nulls / total_rows * 100) if total_rows > 0 else 0,
                'zero_count': piece_count_zeros,
                'status': '✓' if piece_count_nulls == 0 else '✗'
            },
            'material': {
                'null_count': material_nulls,
                'null_percentage': (material_nulls / total_rows * 100) if total_rows > 0 else 0,
                'zero_count': material_zeros,
                'status': '✓' if material_nulls == 0 else '✗'
            },
            'phase': {
                'null_count': phase_nulls,
                'null_percentage': (phase_nulls / total_rows * 100) if total_rows > 0 else 0,
                'zero_count': phase_zeros,
                'status': '✓' if phase_nulls == 0 else '✗'
            }
        }
        
        return derivable_summary
    
    def run_full_validation(self, parquet_file: Path = None) -> Dict:
        """
        Run comprehensive data validation checks.
        Combines null value checks, eval↔wdl consistency, and FEN-derivable field validation.
        """
        print("=" * 80)
        print("RUNNING FULL DATASET VALIDATION")
        print("=" * 80)
        
        # Find most recent file if not specified
        if parquet_file is None:
            parquet_dir = Path(self.parquet_dir)
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if parquet_files:
                parquet_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
                print(f"\nValidating: {parquet_file.name}\n")
        
        # Run all validations
        null_summary = self.validate_null_values(parquet_file)
        consistency_summary = self.validate_eval_wdl_consistency(parquet_file)
        derivable_summary = self.validate_fen_derivable_fields(parquet_file)
        
        # Print NULL value report
        print("1. NULL VALUE DETECTION")
        print("-" * 80)
        if null_summary:
            total_rows = null_summary['total_rows']
            for field, count in null_summary['null_counts'].items():
                pct = null_summary['null_percentages'][field]
                status = "✓" if count == 0 else "⚠"
                print(f"{status} {field:15s}: {count:8d} NULL values ({pct:5.1f}%)")
        
        # Print EVAL ↔ WDL consistency report
        print("\n2. EVAL → WDL CONSISTENCY CHECK")
        print("-" * 80)
        if consistency_summary:
            if consistency_summary['eval_rows_total'] == 0:
                print("⚠ No evaluation data present in dataset (all NULL)")
            else:
                print(f"  Rows with evaluation: {consistency_summary['eval_rows_total']:,}")
                print(f"  Rows with valid WDL:  {consistency_summary['eval_rows_with_valid_wdl']:,}")
                
                if consistency_summary['violations'] == 0:
                    print(f"✓ All eval values are consistent with WDL")
                else:
                    print(f"⚠ Found {consistency_summary['violations']} eval↔wdl inconsistencies")
                    print(f"   ({consistency_summary['violation_percentage']:.1f}% of eval-present rows)")
                    if consistency_summary['violation_details']:
                        print("\n   Top violations:")
                        for v in consistency_summary['violation_details'][:5]:
                            print(f"     Row {v['index']:5d}: eval={v['eval_cp']:6d}cp "
                                  f"(expect wdl={v['wdl_expected']:+d}) → actual wdl={v['wdl_actual']:+d}")
        
        # Print FEN-derivable fields report
        print("\n3. FEN-DERIVABLE FIELDS MUST BE PRESENT")
        print("-" * 80)
        if derivable_summary:
            for field in ['piece_count', 'material', 'phase']:
                info = derivable_summary[field]
                status = info['status']
                print(f"{status} {field:15s}: {info['null_count']:6d} NULL, "
                      f"{info['null_percentage']:5.1f}% null, "
                      f"{info['zero_count']:6d} zeros")
        
        # Final summary
        print("\n" + "=" * 80)
        total_issues = 0
        if null_summary:
            total_issues += sum(null_summary['null_counts'].values())
        if consistency_summary:
            total_issues += consistency_summary['violations']
        if derivable_summary:
            total_issues += sum(f['null_count'] for f in [derivable_summary['piece_count'], 
                                                          derivable_summary['material'], 
                                                          derivable_summary['phase']])
        
        if total_issues == 0:
            print("✓ ALL VALIDATION CHECKS PASSED")
        else:
            print(f"⚠ {total_issues} TOTAL ISSUES FOUND - review data quality")
        print("=" * 80)
        
        # Return summary for programmatic use
        self.validation_results = {
            'null_values': null_summary,
            'eval_wdl_consistency': consistency_summary,
            'fen_derivable_fields': derivable_summary,
            'total_issues': total_issues
        }
        return self.validation_results

if __name__ == "__main__":
    analyzer = DatasetAnalyzer(parquet_dir="v10.0/data/raw", chunk_size=100_000)
    
    # Step 1: Run full validation
    validation_results = analyzer.run_full_validation()
    
    # Step 2: Check for duplicate FENs
    duplicate_check = analyzer.check_duplicate_fens()
    
    # Step 3: Compute statistics
    stats = analyzer.compute_statistics()
    
    # Step 4: Create splits
    splits = analyzer.create_train_val_test_split(train_pct=0.7, val_pct=0.15)
    
    # Step 5: Generate report
    report_path = analyzer.generate_metadata_report(output_dir="v10.0/analysis")
    
    print(f"\n✅ Analysis complete!")
    print(f"   Report: {report_path}")