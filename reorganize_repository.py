#!/usr/bin/env python3
"""
V7P3R Chess AI Repository Reorganization Script
Organizes the repository into v1.0, v2.0, and v3.0 project folders
"""

import os
import shutil
import json
from pathlib import Path

class V7P3RReorganizer:
    """Handles the reorganization of V7P3R Chess AI repository"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.v1_path = self.base_path / "v1.0"
        self.v2_path = self.base_path / "v2.0"
        self.v3_path = self.base_path / "v3.0"
        
        # Files that belong to v1.0 (original implementation)
        self.v1_files = {
            # Root level v1.0 files
            "V7P3RAI_v1.0.spec",
            
            # Original analysis and style files
            "analyze_personal_games.py", 
            "personal_style_analyzer.py",
            "active_game_watcher.py",
            "chess_core.py",
            "stockfish_handler.py",
            "v7p3r_ai.py",
            "v7p3r_training.py",
            "v7p3r_validation.py",
            "play_game.py",
            "train_v7p3r.py",
            "monitor_training.py",
            "extended_validation.py",
            "visualize_training.py",
            
            # Legacy scripts
            "scripts/active_game_watcher.py",
            "scripts/play_game.py",
            "scripts/simulate_games.py",
        }
        
        # Files that belong to v2.0 (current enhanced implementation)
        self.v2_files = {
            # V2.0 specific files
            "v7p3r_gpu_genetic_trainer_clean.py",
            "src/",  # Most of src/ is v2.0
            "scripts/debug_fitness_evaluation.py",
            "scripts/debug_model_diversity.py", 
            "scripts/run_extended_training.py",
            "scripts/enhanced_training_integration.py",
            "scripts/enhanced_training_launcher.py",
            "scripts/test_enhanced_simple.py",
            "scripts/test_enhanced_training.py",
            "scripts/test_opponent_options.py",
            
            # V2.0 documentation
            "docs/README_V2.md",
            "docs/TOURNAMENT_ENGINE_READY.md",
            "docs/TOURNAMENT_STATUS.md",
            "docs/ERROR_FIXES_SUMMARY.md",
            "V2_ENHANCEMENT_SUMMARY.md",
            
            # V2.0 configs
            "enhanced_config.json",
            "enhanced_training_config.json",
            "config.json",
        }
        
        # Shared files (will be copied to all versions)
        self.shared_files = {
            "README.md",
            "requirements.txt",
            "stockfish.exe",
            ".gitignore",
            "docs/V7P3R_CHESS_AI_DESIGN_GUIDE.md",
            "docs/README.md",
        }
        
    def create_version_structure(self):
        """Create the basic structure for each version"""
        for version_path in [self.v1_path, self.v2_path, self.v3_path]:
            for subdir in ["src", "scripts", "models", "data", "docs", "config", "tournament_packages", "reports", "logs"]:
                (version_path / subdir).mkdir(parents=True, exist_ok=True)
                
    def move_v1_files(self):
        """Move v1.0 specific files"""
        print("📁 Moving v1.0 files...")
        
        for file_path in self.v1_files:
            src = self.base_path / file_path
            if src.exists():
                if src.is_file():
                    # Determine destination
                    if file_path.startswith("scripts/"):
                        dst = self.v1_path / file_path
                    else:
                        dst = self.v1_path / src.name
                    
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    print(f"  ✅ Moved {file_path} → v1.0/{dst.relative_to(self.v1_path)}")
                    
    def move_v2_files(self):
        """Move v2.0 specific files"""
        print("📁 Moving v2.0 files...")
        
        for file_path in self.v2_files:
            src = self.base_path / file_path
            if src.exists():
                if src.is_file():
                    # Determine destination
                    if file_path.startswith("scripts/"):
                        dst = self.v2_path / file_path
                    elif file_path.startswith("docs/"):
                        dst = self.v2_path / file_path
                    elif file_path.startswith("src/"):
                        dst = self.v2_path / file_path
                    else:
                        dst = self.v2_path / src.name
                        
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    print(f"  ✅ Moved {file_path} → v2.0/{dst.relative_to(self.v2_path)}")
                elif src.is_dir() and file_path == "src/":
                    # Move entire src directory to v2.0
                    dst = self.v2_path / "src"
                    if dst.exists():
                        shutil.rmtree(str(dst))
                    shutil.move(str(src), str(dst))
                    print(f"  ✅ Moved src/ → v2.0/src/")
                    
    def copy_shared_files(self):
        """Copy shared files to all versions"""
        print("📁 Copying shared files...")
        
        for file_path in self.shared_files:
            src = self.base_path / file_path
            if src.exists():
                for version_path in [self.v1_path, self.v2_path, self.v3_path]:
                    if file_path.startswith("docs/"):
                        dst = version_path / file_path
                    else:
                        dst = version_path / src.name
                        
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_file():
                        shutil.copy2(str(src), str(dst))
                    else:
                        if dst.exists():
                            shutil.rmtree(str(dst))
                        shutil.copytree(str(src), str(dst))
                        
                print(f"  ✅ Copied {file_path} to all versions")
                
    def move_data_directories(self):
        """Move data directories to appropriate versions"""
        print("📁 Moving data directories...")
        
        # V2.0 gets the current models and training data
        data_dirs = {
            "models": self.v2_path / "models",
            "data": self.v2_path / "data", 
            "reports": self.v2_path / "reports",
            "tournament_packages": self.v2_path / "tournament_packages",
            "training_output": self.v2_path / "training_output",
            "validation_results": self.v2_path / "validation_results",
            "simulations": self.v2_path / "simulations",
            "logs": self.v2_path / "logs",
        }
        
        for src_name, dst_path in data_dirs.items():
            src = self.base_path / src_name
            if src.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    shutil.rmtree(str(dst_path))
                shutil.move(str(src), str(dst_path))
                print(f"  ✅ Moved {src_name}/ → v2.0/{src_name}/")
                
        # Create empty data directories for v1.0 and v3.0
        for version_path in [self.v1_path, self.v3_path]:
            for data_dir in ["models", "data", "reports", "logs"]:
                (version_path / data_dir).mkdir(exist_ok=True)
                
    def create_version_configs(self):
        """Create version-specific configuration files"""
        print("📁 Creating version-specific configurations...")
        
        # V1.0 Config
        v1_config = {
            "version": "1.0",
            "description": "Original V7P3R Chess AI with personal style learning",
            "features": [
                "Personal game analysis",
                "Style-based move selection", 
                "Stockfish integration",
                "Basic reinforcement learning"
            ],
            "main_files": {
                "training": "train_v7p3r.py",
                "playing": "play_game.py",
                "analysis": "analyze_personal_games.py"
            }
        }
        
        # V2.0 Config  
        v2_config = {
            "version": "2.0",
            "description": "Enhanced V7P3R with GPU acceleration and genetic algorithms",
            "features": [
                "GPU-accelerated training",
                "Genetic algorithm evolution",
                "Bounty-based fitness system",
                "Tournament engine packaging",
                "Multi-opponent training",
                "Non-deterministic evaluation"
            ],
            "main_files": {
                "training": "v7p3r_gpu_genetic_trainer_clean.py",
                "enhanced_training": "scripts/enhanced_training_integration.py",
                "incremental": "src/training/incremental_trainer.py"
            }
        }
        
        # V3.0 Config (experimental)
        v3_config = {
            "version": "3.0", 
            "description": "Experimental V7P3R approach - TBD",
            "features": [
                "To be determined based on v2.0 learnings",
                "Potential new architecture",
                "Advanced evaluation methods"
            ],
            "status": "experimental",
            "main_files": {}
        }
        
        # Save configs
        for config, version_path in [(v1_config, self.v1_path), (v2_config, self.v2_path), (v3_config, self.v3_path)]:
            config_file = version_path / "config" / "version_info.json"
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"  ✅ Created {config['version']} configuration")
            
    def create_version_readmes(self):
        """Create README files for each version"""
        print("📁 Creating version-specific README files...")
        
        # V1.0 README
        v1_readme = """# V7P3R Chess AI v1.0

Original implementation focusing on personal style learning and basic reinforcement learning.

## Features
- Personal game analysis and style learning
- Stockfish integration for opponent play
- Basic reinforcement learning model
- Style-influenced move selection

## Key Files
- `train_v7p3r.py` - Main training script
- `play_game.py` - Game playing interface
- `analyze_personal_games.py` - Personal style analysis
- `v7p3r_ai.py` - Main AI implementation

## Usage
```bash
# Train the AI
python train_v7p3r.py

# Play games
python play_game.py

# Analyze personal games
python analyze_personal_games.py
```

## Models
Models are stored in `models/` directory.
"""

        # V2.0 README
        v2_readme = """# V7P3R Chess AI v2.0

Enhanced implementation with GPU acceleration, genetic algorithms, and tournament-ready packaging.

## Features
- GPU-accelerated training with CUDA support
- Genetic algorithm evolution with population-based learning
- Bounty-based fitness system for tactical evaluation
- Tournament engine packaging (UCI compatible)
- Multi-opponent training system
- Non-deterministic evaluation for continued learning

## Key Files
- `v7p3r_gpu_genetic_trainer_clean.py` - GPU genetic trainer
- `src/training/incremental_trainer.py` - Incremental training
- `scripts/enhanced_training_integration.py` - Enhanced training system
- `src/core/v7p3r_gpu_model.py` - GPU-optimized neural network

## Usage
```bash
# GPU genetic training
python v7p3r_gpu_genetic_trainer_clean.py

# Enhanced training with multi-opponents
python scripts/enhanced_training_integration.py

# Incremental training from best model
python -m src.training.incremental_trainer
```

## Models
GPU models are stored in `models/` with generation tracking.
Tournament packages in `tournament_packages/`.
"""

        # V3.0 README
        v3_readme = """# V7P3R Chess AI v3.0 (Experimental)

Experimental branch for testing new approaches and architectures.

## Status
**Experimental** - This version is for testing new ideas that may or may not be incorporated into future releases.

## Purpose
This version allows safe experimentation with:
- Alternative neural network architectures
- New training methodologies
- Different evaluation approaches
- Novel chess AI concepts

## Development
This branch can evolve independently of v2.0, allowing for:
- Risk-free experimentation
- Comparison with v2.0 approaches
- Potential integration of successful features

## Note
If experiments prove unsuccessful, development can continue from v2.0 → v4.0, skipping v3.0 entirely.
"""

        # Save README files
        readmes = [
            (v1_readme, self.v1_path / "README.md"),
            (v2_readme, self.v2_path / "README.md"), 
            (v3_readme, self.v3_path / "README.md")
        ]
        
        for content, path in readmes:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Created {path.parent.name}/README.md")
            
    def update_path_references(self):
        """Update path references in moved files"""
        print("📁 Updating path references...")
        
        # This is a placeholder - would need specific file updates
        # For now, create a note about manual updates needed
        
        updates_needed = self.base_path / "PATH_UPDATES_NEEDED.md"
        with open(updates_needed, 'w', encoding='utf-8') as f:
            f.write("""# Path Reference Updates Needed

After reorganization, the following files may need path reference updates:

## V1.0 Files
- Update import paths in Python files
- Update model loading paths
- Update data file references

## V2.0 Files  
- Update import paths in src/ modules
- Update model saving/loading paths
- Update configuration file references
- Update tournament package paths

## V3.0 Files
- Will need fresh configuration once development begins

## Recommended Approach
1. Test each version independently
2. Update imports as needed
3. Verify model loading/saving works
4. Update any hardcoded paths

## Common Path Updates
- `models/` → `v2.0/models/` (for v2.0)
- `src/` → `v2.0/src/` (for v2.0 imports)
- `data/` → `v{version}/data/`
""")
        print(f"  ✅ Created PATH_UPDATES_NEEDED.md with guidance")
        
    def create_root_navigation(self):
        """Create a navigation README for the root directory"""
        nav_readme = """# V7P3R Chess AI - Multi-Version Repository

This repository contains multiple versions of the V7P3R Chess AI, organized for independent development.

## Repository Structure

```
v7p3r-chess-ai/
├── v1.0/          # Original implementation
├── v2.0/          # Current enhanced version  
├── v3.0/          # Experimental branch
└── shared/        # Shared resources (if any)
```

## Version Overview

### v1.0 - Original Implementation
- **Status**: Stable, archived
- **Focus**: Personal style learning, basic RL
- **Use Case**: Reference implementation, simple training

### v2.0 - Enhanced Implementation  
- **Status**: Active development
- **Focus**: GPU acceleration, genetic algorithms, tournament ready
- **Use Case**: Main development branch, tournament play

### v3.0 - Experimental Branch
- **Status**: Experimental
- **Focus**: TBD - new approaches and architectures
- **Use Case**: Risk-free experimentation

## Getting Started

### For v1.0 Development
```bash
cd v1.0/
python train_v7p3r.py
```

### For v2.0 Development (Recommended)
```bash
cd v2.0/
python v7p3r_gpu_genetic_trainer_clean.py
```

### For v3.0 Experimentation
```bash
cd v3.0/
# Set up experimental environment
```

## Development Guidelines

1. **Keep versions independent** - changes in one version don't affect others
2. **Use appropriate version** - v2.0 for serious development, v3.0 for experiments
3. **Cross-pollinate learnings** - successful v3.0 features can be backported to v2.0
4. **Maintain documentation** - update version-specific READMEs

## Migration Path

- **v1.0 → v2.0**: Use existing migration if needed
- **v2.0 → v3.0**: Copy and modify as experiments require
- **v3.0 → v4.0**: If v3.0 proves successful, could become basis for v4.0

## Current Status

- **v1.0**: Organized and preserved
- **v2.0**: Active, contains latest enhancements (genetic algorithms, GPU training, tournament packages)
- **v3.0**: Ready for experimentation

Choose your version and start developing!
"""

        with open(self.base_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(nav_readme)
        print("  ✅ Created root navigation README.md")
        
    def cleanup_root(self):
        """Clean up remaining files in root"""
        print("📁 Cleaning up root directory...")
        
        # Files that should be removed from root after migration
        cleanup_files = [
            "active_game.pgn",
            "config/",
            "game_history/",
            "images/", 
            "build/",
            "builds/",
            "dist/",
            "__pycache__/"
        ]
        
        for item in cleanup_files:
            path = self.base_path / item
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(str(path))
                print(f"  ✅ Cleaned up {item}")
                
    def reorganize(self):
        """Run the complete reorganization"""
        print("🚀 Starting V7P3R Chess AI Repository Reorganization")
        print("=" * 60)
        
        try:
            self.create_version_structure()
            self.move_v1_files()
            self.move_v2_files() 
            self.copy_shared_files()
            self.move_data_directories()
            self.create_version_configs()
            self.create_version_readmes()
            self.update_path_references()
            self.create_root_navigation()
            self.cleanup_root()
            
            print("\n" + "=" * 60)
            print("✅ REORGANIZATION COMPLETE!")
            print("=" * 60)
            print("\n📋 Next Steps:")
            print("1. Test each version independently")
            print("2. Update any remaining path references") 
            print("3. Verify model loading/saving in each version")
            print("4. Start v3.0 development when ready")
            print("\n🎯 Current Status:")
            print("• v1.0: Ready for legacy use")
            print("• v2.0: Ready for continued development") 
            print("• v3.0: Ready for experimentation")
            
        except Exception as e:
            print(f"\n❌ Error during reorganization: {e}")
            print("Check the paths and try again")
            return False
            
        return True

def main():
    """Main entry point"""
    base_path = Path(__file__).parent
    reorganizer = V7P3RReorganizer(base_path)
    
    print("V7P3R Chess AI Repository Reorganization")
    print("This will organize the repository into v1.0, v2.0, and v3.0 folders")
    print(f"Base path: {base_path}")
    
    response = input("\nProceed with reorganization? (y/N): ")
    if response.lower() != 'y':
        print("Reorganization cancelled")
        return
        
    success = reorganizer.reorganize()
    if success:
        print("\n🎉 Repository successfully reorganized!")
    else:
        print("\n❌ Reorganization failed - check errors above")

if __name__ == "__main__":
    main()
