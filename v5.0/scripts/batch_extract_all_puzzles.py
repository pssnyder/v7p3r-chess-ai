#!/usr/bin/env python3
"""
Batch Puzzle Extraction Script
Processes all V7P3R puzzle analysis files from analysis_results directory
Combines them into a single training dataset with version tracking

Author: V7P3R AI Training Pipeline
Version: 5.0
Created: 2026-05-06
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import sys
from datetime import datetime


class BatchPuzzleExtractor:
    """Orchestrates extraction from multiple puzzle analysis files"""
    
    def __init__(self, analysis_dir: Path, output_dir: Path, extractor_script: Path):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.extractor_script = extractor_script
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_puzzles': 0,
            'total_positions': 0,
            'versions_found': set(),
            'file_results': []
        }
    
    def find_puzzle_files(self) -> List[Path]:
        """Find all puzzle analysis JSON files in the analysis directory"""
        puzzle_files = []
        
        # Find enhanced_sequence_analysis files
        enhanced_files = list(self.analysis_dir.glob("*enhanced_sequence_analysis*.json"))
        puzzle_files.extend(enhanced_files)
        
        # Find puzzle_results files
        results_files = list(self.analysis_dir.glob("puzzle_results*.json"))
        puzzle_files.extend(results_files)
        
        # Sort by version and date
        puzzle_files.sort()
        
        return puzzle_files
    
    def extract_version_from_filename(self, filename: str) -> str:
        """Extract V7P3R version from filename"""
        # Examples:
        # V7P3R_v17_1_1_enhanced_sequence_analysis_20251125_220843.json -> v17.1.1
        # puzzle_results_v18_3_20260416_183756.json -> v18.3
        
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.startswith('v') and len(part) > 1:
                # Try to extract version components
                version_parts = []
                j = i
                while j < len(parts):
                    if parts[j].startswith('v'):
                        version_parts.append(parts[j])
                    elif parts[j].isdigit() and len(version_parts) > 0:
                        version_parts.append(parts[j])
                    else:
                        break
                    j += 1
                    
                    # Stop if we hit a date-like pattern or other marker
                    if j < len(parts) and (len(parts[j]) == 8 or parts[j] in ['enhanced', 'sequence', 'analysis']):
                        break
                
                # Reconstruct version
                if version_parts:
                    version = '.'.join(version_parts).replace('v.', 'v')
                    # Clean up common patterns
                    version = version.replace('..', '.')
                    return version
        
        return "unknown"
    
    def process_single_file(self, puzzle_file: Path) -> Dict[str, Any]:
        """Process a single puzzle analysis file"""
        filename = puzzle_file.name
        version = self.extract_version_from_filename(filename)
        
        # Create version-specific output file
        output_file = self.output_dir / f"extracted_{puzzle_file.stem}.jsonl"
        
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"Version: {version}")
        print(f"Output: {output_file.name}")
        print(f"{'='*80}")
        
        # Build command to run extractor
        cmd = [
            sys.executable,
            str(self.extractor_script),
            '--input', str(puzzle_file),
            '--output', str(output_file),
            '--engine-version', version
        ]
        
        try:
            # Run extraction
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            if result.returncode == 0:
                # Count extracted positions
                position_count = 0
                puzzle_count = 0
                
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                position_count += 1
                                # Track unique puzzles by checking metadata
                                try:
                                    record = json.loads(line)
                                    if 'metadata' in record:
                                        puzzle_count += 1
                                except:
                                    pass
                
                # Update stats
                self.stats['files_processed'] += 1
                self.stats['total_positions'] += position_count
                self.stats['versions_found'].add(version)
                
                file_result = {
                    'filename': filename,
                    'version': version,
                    'status': 'success',
                    'positions': position_count,
                    'output_file': str(output_file)
                }
                
                print(f"✓ SUCCESS: Extracted {position_count} positions")
                
                return file_result
            else:
                # Extraction failed
                self.stats['files_failed'] += 1
                
                file_result = {
                    'filename': filename,
                    'version': version,
                    'status': 'failed',
                    'error': result.stderr,
                    'positions': 0
                }
                
                print(f"✗ FAILED: {result.stderr}")
                
                return file_result
                
        except subprocess.TimeoutExpired:
            self.stats['files_failed'] += 1
            
            file_result = {
                'filename': filename,
                'version': version,
                'status': 'timeout',
                'error': 'Extraction timed out after 5 minutes',
                'positions': 0
            }
            
            print(f"✗ TIMEOUT: Processing took too long")
            
            return file_result
        
        except Exception as e:
            self.stats['files_failed'] += 1
            
            file_result = {
                'filename': filename,
                'version': version,
                'status': 'error',
                'error': str(e),
                'positions': 0
            }
            
            print(f"✗ ERROR: {str(e)}")
            
            return file_result
    
    def combine_all_extractions(self, combined_output: Path):
        """Combine all individual extraction files into one master dataset"""
        print(f"\n{'='*80}")
        print("Combining all extracted positions into master dataset...")
        print(f"{'='*80}")
        
        total_combined = 0
        
        # Find all extraction output files
        extraction_files = list(self.output_dir.glob("extracted_*.jsonl"))
        
        with open(combined_output, 'w', encoding='utf-8') as outfile:
            for ext_file in sorted(extraction_files):
                if ext_file.exists():
                    with open(ext_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():
                                outfile.write(line)
                                total_combined += 1
        
        print(f"✓ Combined {total_combined} positions from {len(extraction_files)} files")
        print(f"✓ Master dataset: {combined_output}")
        
        return total_combined
    
    def save_batch_stats(self, stats_file: Path):
        """Save batch processing statistics"""
        stats_data = {
            'batch_extraction_timestamp': datetime.now().isoformat(),
            'files_processed': self.stats['files_processed'],
            'files_failed': self.stats['files_failed'],
            'total_positions': self.stats['total_positions'],
            'versions_found': sorted(list(self.stats['versions_found'])),
            'file_results': self.stats['file_results']
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"\n✓ Batch statistics saved: {stats_file}")
    
    def run_batch_extraction(self) -> Dict[str, Any]:
        """Main batch extraction workflow"""
        print("="*80)
        print("V7P3R AI v5.0 - Batch Puzzle Extraction")
        print("="*80)
        
        # Find all puzzle files
        puzzle_files = self.find_puzzle_files()
        
        print(f"\nFound {len(puzzle_files)} puzzle analysis files")
        print(f"Analysis directory: {self.analysis_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Process each file
        for puzzle_file in puzzle_files:
            file_result = self.process_single_file(puzzle_file)
            self.stats['file_results'].append(file_result)
        
        # Combine all extractions
        combined_output = self.output_dir / "all_puzzles_combined.jsonl"
        total_combined = self.combine_all_extractions(combined_output)
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH EXTRACTION SUMMARY")
        print("="*80)
        print(f"Files Processed: {self.stats['files_processed']}")
        print(f"Files Failed: {self.stats['files_failed']}")
        print(f"Total Positions: {total_combined}")
        print(f"V7P3R Versions: {', '.join(sorted(self.stats['versions_found']))}")
        print("="*80)
        
        # Save statistics
        stats_file = self.output_dir / "batch_extraction_stats.json"
        self.save_batch_stats(stats_file)
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract all V7P3R puzzle analysis files'
    )
    parser.add_argument(
        '--analysis-dir',
        type=Path,
        default=Path(r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\analysis_results'),
        help='Directory containing puzzle analysis JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/puzzles/batch_extracted'),
        help='Output directory for extracted puzzle positions'
    )
    parser.add_argument(
        '--extractor-script',
        type=Path,
        default=Path('scripts/extract_puzzle_results.py'),
        help='Path to extract_puzzle_results.py script'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.analysis_dir.exists():
        print(f"ERROR: Analysis directory not found: {args.analysis_dir}")
        sys.exit(1)
    
    if not args.extractor_script.exists():
        print(f"ERROR: Extractor script not found: {args.extractor_script}")
        sys.exit(1)
    
    # Run batch extraction
    extractor = BatchPuzzleExtractor(
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        extractor_script=args.extractor_script
    )
    
    stats = extractor.run_batch_extraction()
    
    # Exit with appropriate code
    if stats['files_failed'] > 0:
        print(f"\n⚠ Warning: {stats['files_failed']} files failed to process")
        sys.exit(1)
    else:
        print(f"\n✓ All files processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
