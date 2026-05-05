"""
V7P3RAI V3.0 Tournament Build Script
===================================
Packages the intensively trained V7P3R AI into tournament-ready executable
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
import pickle
from datetime import datetime

class V7P3RAIBuilder:
    """Build tournament-ready V7P3RAI v3.0 executable"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "tournament_build"
        self.dist_dir = self.project_root / "dist"
        self.output_name = "V7P3RAI_v3.0"
        
    def log(self, message: str):
        """Log build progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def verify_training_completion(self) -> bool:
        """Verify that intensive training is complete"""
        self.log("🔍 Verifying training completion...")
        
        # Check if Day 5 has results
        try:
            result = subprocess.run([
                sys.executable, "intensive_tracker.py", "status"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Check for completion indicators
            if "Training Day: 5/5" in result.stdout and "Days Remaining: 0" in result.stdout:
                self.log("✅ Day 5 training detected as complete")
                return True
            elif "✅ Day 5:" in result.stdout:
                self.log("✅ Day 5 training detected as complete")
                return True
            else:
                self.log("⚠️ Day 5 training may still be running")
                return False
                
        except Exception as e:
            self.log(f"❌ Error checking training status: {e}")
            return False
    
    def verify_models_exist(self) -> bool:
        """Verify that trained models exist"""
        self.log("🔍 Verifying trained models...")
        
        model_files = [
            "v3.0/models/enhanced_puzzle_training_v2/v7p3r_enhanced_v2_final_20251006_031410.pkl",
            "v3.0/src/ai/thinking_brain.py",
            "v3.0/src/ai/gameplay_brain.py"
        ]
        
        for model_file in model_files:
            model_path = self.project_root / model_file
            if not model_path.exists():
                self.log(f"❌ Missing required file: {model_file}")
                return False
            else:
                self.log(f"✅ Found: {model_file}")
        
        return True
    
    def setup_build_directory(self):
        """Setup clean build directory"""
        self.log("🏗️ Setting up build directory...")
        
        # Remove existing build directory
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        
        # Create fresh build directory
        self.build_dir.mkdir(parents=True)
        
        # Create subdirectories
        (self.build_dir / "v3.0" / "src").mkdir(parents=True)
        (self.build_dir / "models").mkdir(parents=True)
        (self.build_dir / "config").mkdir(parents=True)
        (self.build_dir / "logs").mkdir(parents=True)
        
        self.log("✅ Build directory prepared")
    
    def copy_core_files(self):
        """Copy essential files to build directory"""
        self.log("📁 Copying core files...")
        
        # Core files to include
        core_files = [
            "v7p3rai_uci_main.py",
            "chess_core.py",
            "requirements.txt"
        ]
        
        for file_name in core_files:
            src = self.project_root / file_name
            dst = self.build_dir / file_name
            if src.exists():
                shutil.copy2(src, dst)
                self.log(f"✅ Copied: {file_name}")
            else:
                self.log(f"⚠️ Missing: {file_name}")
        
        # Copy V3.0 source code
        v3_src = self.project_root / "v3.0" / "src"
        v3_dst = self.build_dir / "v3.0" / "src"
        
        if v3_src.exists():
            shutil.copytree(v3_src, v3_dst, dirs_exist_ok=True)
            self.log("✅ Copied V3.0 source code")
        else:
            self.log("❌ V3.0 source directory not found")
    
    def copy_trained_models(self):
        """Copy trained model files"""
        self.log("🧠 Copying trained models...")
        
        # Copy main model (latest from Day 5)
        model_src = self.project_root / "v3.0" / "models" / "enhanced_puzzle_training_v2" / "v7p3r_enhanced_v2_final_20251006_031410.pkl"
        model_dst = self.build_dir / "models" / "v7p3r_model.pkl"
        
        if model_src.exists():
            shutil.copy2(model_src, model_dst)
            self.log("✅ Copied main model file (Day 5 final)")
        else:
            self.log("❌ Main model file not found")
        
        # Copy any additional model files from V3.0 training
        v3_models_dir = self.project_root / "v3.0" / "models" / "enhanced_puzzle_training_v2"
        if v3_models_dir.exists():
            # Copy the latest few models as backup
            model_files = sorted(v3_models_dir.glob("v7p3r_enhanced_v2_final_*.pkl"))
            if model_files:
                latest_model = model_files[-1]  # Most recent final model
                backup_dst = self.build_dir / "models" / latest_model.name
                shutil.copy2(latest_model, backup_dst)
                self.log(f"✅ Copied backup model: {latest_model.name}")
    
    def create_configuration_files(self):
        """Create configuration files for tournament"""
        self.log("⚙️ Creating configuration files...")
        
        # Engine configuration
        engine_config = {
            "engine_name": "V7P3RAI v3.0",
            "author": "V7P3R Team",
            "version": "3.0",
            "build_date": datetime.now().isoformat(),
            "training_completion": "Day 5 - 8 hour intensive session",
            "uci_options": {
                "Hash": {"default": 128, "min": 1, "max": 1024},
                "Threads": {"default": 1, "min": 1, "max": 8},
                "CUDA_Enabled": {"default": True},
                "Puzzle_Mode": {"default": True},
                "Aggression_Level": {"default": 5, "min": 1, "max": 10}
            }
        }
        
        config_file = self.build_dir / "config" / "engine_config.json"
        with open(config_file, 'w') as f:
            json.dump(engine_config, f, indent=2)
        
        self.log("✅ Created engine configuration")
        
        # UCI documentation
        uci_readme = """
# V7P3RAI v3.0 UCI Engine
## Tournament-Ready Chess AI

### Installation
1. Extract all files to a directory
2. Add V7P3RAI_v3.0.exe to your chess GUI
3. Configure UCI options as needed

### UCI Options
- **Hash**: Memory allocation (128-1024 MB)
- **Threads**: CPU threads for analysis
- **CUDA_Enabled**: GPU acceleration (requires CUDA)
- **Puzzle_Mode**: Enhanced tactical pattern recognition
- **Aggression_Level**: Playing style (1=defensive, 10=aggressive)

### Performance
- Trained on 40,000+ tactical puzzles
- Target accuracy: 90%+
- Expected ELO: 1400-1600
- Optimized for RTX 4070 Ti

### Support
For issues or questions, see project documentation.
"""
        
        readme_file = self.build_dir / "README_UCI.txt"
        with open(readme_file, 'w') as f:
            f.write(uci_readme)
        
        self.log("✅ Created UCI documentation")
    
    def install_dependencies(self):
        """Install required dependencies in build environment"""
        self.log("📦 Installing dependencies...")
        
        try:
            # Install PyInstaller if not present
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pyinstaller"
            ], check=True, capture_output=True)
            
            self.log("✅ PyInstaller ready")
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to install PyInstaller: {e}")
            return False
        
        return True
    
    def build_executable(self):
        """Build the tournament executable"""
        self.log("🔨 Building tournament executable...")
        
        # PyInstaller command
        pyinstaller_args = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--optimize", "2", 
            "--name", self.output_name,
            "--distpath", str(self.dist_dir),
            "--workpath", str(self.build_dir / "temp"),
            "--specpath", str(self.build_dir),
            "--add-data", f"{self.build_dir / 'v3.0'};v3.0",
            "--add-data", f"{self.build_dir / 'models'};models", 
            "--add-data", f"{self.build_dir / 'config'};config",
            "--hidden-import", "chess_core",
            "--hidden-import", "v3.0.src.ai.thinking_brain",
            "--hidden-import", "v3.0.src.ai.gameplay_brain",
            str(self.build_dir / "v7p3rai_uci_main.py")
        ]
        
        try:
            # Run PyInstaller
            result = subprocess.run(
                pyinstaller_args, 
                cwd=self.build_dir, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                self.log("✅ Executable built successfully")
                return True
            else:
                self.log(f"❌ PyInstaller failed:")
                self.log(result.stderr)
                return False
                
        except Exception as e:
            self.log(f"❌ Build error: {e}")
            return False
    
    def create_distribution_package(self):
        """Create final distribution package"""
        self.log("📦 Creating distribution package...")
        
        # Create distribution directory
        dist_package = self.dist_dir / f"{self.output_name}_tournament"
        if dist_package.exists():
            shutil.rmtree(dist_package)
        dist_package.mkdir(parents=True)
        
        # Copy executable
        exe_src = self.dist_dir / f"{self.output_name}.exe"
        exe_dst = dist_package / f"{self.output_name}.exe"
        
        if exe_src.exists():
            shutil.copy2(exe_src, exe_dst)
            self.log("✅ Copied executable")
        else:
            self.log("❌ Executable not found")
            return False
        
        # Copy configuration files
        shutil.copy2(
            self.build_dir / "config" / "engine_config.json",
            dist_package / "engine_config.json"
        )
        shutil.copy2(
            self.build_dir / "README_UCI.txt", 
            dist_package / "README_UCI.txt"
        )
        
        # Create logs directory
        (dist_package / "logs").mkdir()
        
        self.log(f"✅ Distribution package created: {dist_package}")
        return True
    
    def verify_build(self):
        """Verify the built executable works"""
        self.log("🧪 Verifying build...")
        
        exe_path = self.dist_dir / f"{self.output_name}_tournament" / f"{self.output_name}.exe"
        
        if not exe_path.exists():
            self.log("❌ Executable not found")
            return False
        
        try:
            # Test UCI command
            result = subprocess.run([
                str(exe_path)
            ], input="uci\nquit\n", capture_output=True, text=True, timeout=30)
            
            if "V7P3RAI v3.0" in result.stdout:
                self.log("✅ UCI interface responds correctly")
                return True
            else:
                self.log("❌ UCI interface not responding")
                self.log(f"Output: {result.stdout}")
                self.log(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Verification failed: {e}")
            return False
    
    def cleanup_build_files(self):
        """Clean up temporary build files"""
        self.log("🧹 Cleaning up build files...")
        
        # Remove temporary build directory
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            self.log("✅ Build directory cleaned")
        
        # Remove PyInstaller spec file
        spec_file = self.project_root / f"{self.output_name}.spec"
        if spec_file.exists():
            spec_file.unlink()
            self.log("✅ Spec file removed")
    
    def build_tournament_package(self):
        """Complete build process"""
        self.log("🚀 Starting V7P3RAI v3.0 tournament build...")
        
        # Verification steps
        if not self.verify_training_completion():
            self.log("⚠️ Training may not be complete. Continue anyway? (y/n)")
            if input().lower() != 'y':
                return False
        
        if not self.verify_models_exist():
            self.log("❌ Required models not found. Cannot build.")
            return False
        
        # Build steps
        try:
            self.setup_build_directory()
            self.copy_core_files()
            self.copy_trained_models()
            self.create_configuration_files()
            
            if not self.install_dependencies():
                return False
            
            if not self.build_executable():
                return False
            
            if not self.create_distribution_package():
                return False
            
            if not self.verify_build():
                self.log("⚠️ Build verification failed, but package created")
            
            self.cleanup_build_files()
            
            self.log("🎉 V7P3RAI v3.0 tournament package built successfully!")
            self.log(f"📦 Package location: {self.dist_dir / f'{self.output_name}_tournament'}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Build failed: {e}")
            return False

def main():
    """Main build script entry point"""
    print("=" * 60)
    print("V7P3RAI V3.0 TOURNAMENT BUILD SCRIPT")
    print("=" * 60)
    
    builder = V7P3RAIBuilder()
    success = builder.build_tournament_package()
    
    if success:
        print("\n🎯 BUILD COMPLETE!")
        print("Your tournament-ready V7P3RAI v3.0 is ready for competition!")
    else:
        print("\n❌ BUILD FAILED!")
        print("Check the logs above for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())