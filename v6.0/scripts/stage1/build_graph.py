"""
Transposition Graph Builder - V7P3R AI v6.0

Builds a graph connecting similar chess positions for transposition learning.

Nodes = Positions (with Zobrist hash as ID)
Edges = Similarity links (based on feature distance or structural similarity)
"""

import json
import sys
import chess
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Import Zobrist hashing
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from zobrist_hashing import hash_fen, get_hasher


class TranspositionGraphBuilder:
    """Build transposition graph from filtered positions."""
    
    def __init__(self, good_positions_path: str, output_path: str):
        """
        Args:
            good_positions_path: Path to filtered good positions
            output_path: Output path for graph pickle
        """
        self.input_path = Path(good_positions_path)
        self.output_path = Path(output_path)
        
        # Graph structure (adjacency list)
        # {zobrist_hash: {'fen': str, 'features': dict, 'neighbors': set()}}
        self.graph = {}
        
        # Position index for fast lookup
        # {zobrist_hash: position_data}
        self.position_index = {}
        
        # Statistics
        self.stats = defaultdict(int)
    
    def build_graph(self, max_positions: int = None):
        """
        Build transposition graph from filtered positions.
        
        Args:
            max_positions: Limit number of positions (for testing)
        """
        print("="*60)
        print("BUILDING TRANSPOSITION GRAPH - V7P3R AI v6.0")
        print("="*60)
        print(f"\nInput: {self.input_path}")
        print()
        
        # Phase 1: Index all positions
        print("Phase 1: Indexing positions by Zobrist hash...")
        self._index_positions(max_positions)
        
        # Phase 2: Find similar positions (OPTIMIZED - small sample)
        print("\nPhase 2: Finding similar positions...")
        self._find_similar_positions(sample_size=1000, comparison_pool=50000)
        
        # Phase 3: Save graph
        print("\nPhase 3: Saving graph...")
        self._save_graph()
        
        # Report
        self.print_report()
    
    def _index_positions(self, max_positions: int = None):
        """
        Phase 1: Index all positions by Zobrist hash.
        
        Detects duplicate positions across different games/puzzles.
        """
        hasher = get_hasher()
        
        with open(self.input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract FEN
                    fen = None
                    if 'position' in record and 'fen' in record['position']:
                        fen = record['position']['fen']
                    elif 'features' in record and 'F001_position_fen' in record['features']:
                        fen = record['features']['F001_position_fen']
                    
                    if not fen:
                        self.stats['no_fen'] += 1
                        continue
                    
                    # Compute Zobrist hash
                    try:
                        pos_hash = hasher.hash_fen(fen)
                    except Exception as e:
                        self.stats['hash_errors'] += 1
                        continue
                    
                    # Check for duplicate
                    if pos_hash in self.position_index:
                        self.stats['duplicates'] += 1
                        continue
                    
                    # Store position
                    self.position_index[pos_hash] = {
                        'fen': fen,
                        'features': record.get('features', {}),
                        'source': record.get('source', 'unknown'),
                        'grade': record.get('stockfish_analysis', {}).get('grade', 0)
                    }
                    
                    # Initialize graph node
                    self.graph[pos_hash] = {
                        'fen': fen,
                        'features': record.get('features', {}),
                        'neighbors': set()  # Will be populated in Phase 2
                    }
                    
                    self.stats['positions_indexed'] += 1
                    
                    # Progress update
                    if line_num % 100000 == 0:
                        print(f"  Indexed: {line_num:,} records, {self.stats['positions_indexed']:,} unique positions")
                    
                    # Stop at max if set
                    if max_positions and self.stats['positions_indexed'] >= max_positions:
                        break
                
                except Exception as e:
                    self.stats['errors'] += 1
                    if self.stats['errors'] < 10:
                        print(f"  Error on line {line_num}: {e}")
        
        print(f"\n✅ Indexed {self.stats['positions_indexed']:,} unique positions")
        print(f"   Duplicates found: {self.stats['duplicates']:,}")
    
    def _find_similar_positions(self, sample_size: int = 1000, comparison_pool: int = 50000):
        """
        Phase 2: Find similar positions and create edges.
        
        OPTIMIZED VERSION: 
        - Only builds graph for small sample (1000 positions)
        - Compares against limited pool (50k positions)
        - Full graph requires FAISS/Annoy (future enhancement)
        
        Similarity metric: Number of shared tactical features
        (hanging pieces, pins, forks, etc.)
        """
        print(f"  Building graph for {sample_size} sampled positions...")
        print(f"  (Comparing against pool of {comparison_pool} positions)")
        print(f"  Note: Full 5.6M graph requires FAISS - this is a representative sample")
        
        # Get tactical feature names
        tactical_features = [
            'F040_white_has_hanging_pieces', 'F040_black_has_hanging_pieces',
            'F044_white_has_fork_threat', 'F044_black_has_fork_threat',
            'F045_white_has_pin', 'F045_black_has_pin',
            'F046_white_has_skewer', 'F046_black_has_skewer',
            'F047_white_has_discovered_attack', 'F047_black_has_discovered_attack',
            'F012_white_king_under_attack', 'F012_black_king_under_attack',
            'F020_white_has_passed_pawns', 'F020_black_has_passed_pawns',
        ]
        
        # Sample positions for graph nodes
        all_hashes = list(self.graph.keys())
        
        import random
        random.seed(42)
        
        # Sample nodes to build graph for
        if len(all_hashes) <= sample_size:
            sample_hashes = all_hashes
        else:
            sample_hashes = random.sample(all_hashes, sample_size)
        
        # Sample comparison pool
        if len(all_hashes) <= comparison_pool:
            pool_hashes = all_hashes
        else:
            pool_hashes = random.sample(all_hashes, comparison_pool)
        
        print(f"  Graph nodes: {len(sample_hashes):,}")
        print(f"  Comparison pool: {len(pool_hashes):,}")
        
        # For each sampled position, find K similar positions
        for i, hash1 in enumerate(sample_hashes):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(sample_hashes)} nodes...")
            
            features1 = self.graph[hash1]['features']
            
            # Extract tactical features
            tactical1 = self._extract_tactical_features(features1, tactical_features)
            
            # Compare to pool positions (much faster than all 5.6M)
            similarities = []
            
            for hash2 in pool_hashes:
                if hash1 == hash2:
                    continue
                
                features2 = self.graph[hash2]['features']
                tactical2 = self._extract_tactical_features(features2, tactical_features)
                
                # Similarity = number of shared tactical features
                similarity = sum(1 for f1, f2 in zip(tactical1, tactical2) if f1 == f2 and f1 == True)
                
                if similarity > 0:
                    similarities.append((hash2, similarity))
            
            # Keep top K neighbors
            K = 10
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:K]
            
            # Add edges
            for neighbor_hash, sim_score in top_k:
                self.graph[hash1]['neighbors'].add(neighbor_hash)
                self.graph[neighbor_hash]['neighbors'].add(hash1)  # Undirected
                self.stats['edges_created'] += 1
            
            # Progress
            if (i + 1) % 1000 == 0:
                print(f"  Processed: {i+1:,} / {len(sample_hashes):,} positions")
        
        print(f"\n✅ Created {self.stats['edges_created']:,} edges")
        
        # Cleanup: Remove positions not in the sample graph
        print(f"\nCleaning up: Keeping only {len(sample_hashes):,} sampled positions...")
        positions_to_keep = set(sample_hashes)
        
        # Also keep neighbors that were added to the graph
        for hash_val in sample_hashes:
            positions_to_keep.update(self.graph[hash_val]['neighbors'])
        
        # Remove positions not in the final graph
        all_positions = list(self.graph.keys())
        for hash_val in all_positions:
            if hash_val not in positions_to_keep:
                del self.graph[hash_val]
        
        print(f"✅ Final graph size: {len(self.graph):,} positions")
    
    def _extract_tactical_features(self, features: dict, feature_names: list) -> list:
        """Extract tactical feature values."""
        return [features.get(name, False) for name in feature_names]
    
    def _save_graph(self):
        """Save graph to pickle file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        print(f"✅ Graph saved to: {self.output_path}")
    
    def print_report(self):
        """Print graph building report."""
        print("\n" + "="*60)
        print("GRAPH BUILDING REPORT")
        print("="*60)
        
        print("\n⚠️  SAMPLE GRAPH (Optimized for Speed)")
        print("-" * 60)
        print("This is a representative sample graph.")
        print("Full 5.6M graph requires FAISS/Annoy (future enhancement).")
        print("Sample is sufficient for initial v6.0 training & validation.")
        
        print("\n📊 GRAPH STATISTICS")
        print("-" * 60)
        print(f"Total positions indexed:         {self.stats['positions_indexed']:,}")
        print(f"Graph nodes (in sample):         {len(self.graph):,}")
        print(f"Total edges (similarity links):  {self.stats['edges_created']:,}")
        
        # Calculate average degree
        total_degree = sum(len(node['neighbors']) for node in self.graph.values())
        avg_degree = total_degree / len(self.graph) if len(self.graph) > 0 else 0
        print(f"Average node degree:             {avg_degree:.2f}")
        
        # Find most connected positions
        most_connected = sorted(
            self.graph.items(),
            key=lambda x: len(x[1]['neighbors']),
            reverse=True
        )[:5]
        
        print("\n🔗 MOST CONNECTED POSITIONS")
        print("-" * 60)
        for pos_hash, data in most_connected:
            fen = data['fen']
            degree = len(data['neighbors'])
            print(f"  {fen[:40]}... → {degree} neighbors")
        
        print("\n💾 OUTPUT")
        print("-" * 60)
        print(f"Graph file: {self.output_path}")
        print(f"Size: {self.output_path.stat().st_size / 1e6:.2f} MB")
        
        print("\n" + "="*60)


def main():
    """Main execution."""
    
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / "data" / "stage1" / "good_positions.jsonl"
    output_file = base_dir / "data" / "stage1" / "transposition_graph.pkl"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run filter_dataset.py first!")
        return 1
    
    # Build graph
    builder = TranspositionGraphBuilder(
        good_positions_path=str(input_file),
        output_path=str(output_file)
    )
    
    # For testing: limit to 50k positions
    # For production: remove max_positions limit
    builder.build_graph(max_positions=None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
