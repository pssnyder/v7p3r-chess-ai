#!/usr/bin/env python3
"""
Incremental GPU Training for V7P3R Chess AI v2.0
Supports loading previous best models and continuing training with modified fitness functions
"""

import os
import json
import time
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List

from v7p3r_gpu_genetic_trainer_clean import V7P3RGPUGeneticTrainer
from v7p3r_gpu_model import V7P3RGPU_LSTM

class V7P3RIncrementalTrainer(V7P3RGPUGeneticTrainer):
    """
    Enhanced GPU trainer that supports incremental training.
    Can load previous best models and continue training with modified parameters.
    """
    
    def __init__(self, config: Dict[str, Any], load_previous_best: bool = True):
        super().__init__(config)
        self.load_previous_best = load_previous_best
        self.previous_model_path = None
        
    def find_latest_best_model(self) -> Optional[str]:
        """Find the most recent best model file"""
        model_pattern = "models/best_gpu_model_gen_*.pth"
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            return None
        
        # Sort by generation number
        def extract_gen_number(filepath):
            try:
                # Extract generation number from filename
                base = os.path.basename(filepath)
                gen_part = base.split('gen_')[1].split('.')[0]
                return int(gen_part)
            except:
                return -1
        
        latest_model = max(model_files, key=extract_gen_number)
        return latest_model
    
    def load_best_model_into_population(self, model_path: str, num_copies: int = 4) -> List[V7P3RGPU_LSTM]:
        """Load the best model and create copies for seeding the population"""
        print(f"Loading previous best model from: {model_path}")
        
        try:
            best_model = V7P3RGPU_LSTM.load_model(model_path, self.device)
            
            # Create multiple copies with slight variations
            seeded_models = []
            
            # Add the original best model
            seeded_models.append(best_model)
            
            # Create copies with small mutations
            for i in range(num_copies - 1):
                model_copy = V7P3RGPU_LSTM.load_model(model_path, self.device)
                # Apply small mutation to create diversity
                model_copy.mutate(mutation_strength=0.01)  # Very small mutations
                seeded_models.append(model_copy)
            
            print(f"Successfully loaded and created {len(seeded_models)} seeded models")
            return seeded_models
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return []
    
    def create_incremental_population(self, population_size: int) -> List[V7P3RGPU_LSTM]:
        """Create population seeded with previous best models"""
        population = []
        
        # Try to load previous best model if requested
        if self.load_previous_best:
            latest_model_path = self.find_latest_best_model()
            
            if latest_model_path:
                self.previous_model_path = latest_model_path
                print(f"Found previous best model: {latest_model_path}")
                
                # Seed 25% of population with previous best model variants
                num_seeded = max(1, population_size // 4)
                seeded_models = self.load_best_model_into_population(latest_model_path, num_seeded)
                population.extend(seeded_models)
                
                print(f"Seeded {len(seeded_models)}/{population_size} models from previous training")
            else:
                print("No previous best model found, starting fresh")
        
        # Fill remaining population with random models
        remaining = population_size - len(population)
        if remaining > 0:
            print(f"Creating {remaining} new random models...")
            random_models = super().create_random_population(remaining)
            population.extend(random_models)
        
        return population
    
    def create_random_population(self, population_size: int) -> List[V7P3RGPU_LSTM]:
        """Override to use incremental population creation"""
        return self.create_incremental_population(population_size)
    
    def train_with_modified_fitness(self, num_generations: int, 
                                   bounty_modifications: Optional[Dict[str, float]] = None,
                                   fitness_modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train with modified fitness function while building on previous models.
        
        Args:
            num_generations: Number of generations to train
            bounty_modifications: Dictionary of bounty weight modifications
            fitness_modifications: Dictionary of other fitness modifications
        """
        
        print(f"Starting incremental training for {num_generations} generations...")
        
        if self.previous_model_path:
            print(f"Building upon previous model: {self.previous_model_path}")
        
        # Apply bounty modifications if provided
        if bounty_modifications:
            print("Applying bounty modifications:")
            for param, value in bounty_modifications.items():
                print(f"  {param}: {value}")
                # Apply modifications to bounty system
                if hasattr(self.bounty_system, param):
                    setattr(self.bounty_system, param, value)
        
        # Apply other fitness modifications
        if fitness_modifications:
            print("Applying fitness modifications:")
            for param, value in fitness_modifications.items():
                print(f"  {param}: {value}")
                if param in self.config:
                    self.config[param] = value
        
        # Run normal training
        return self.train(num_generations)
    
    def save_incremental_report(self, report: Dict[str, Any], 
                               bounty_modifications: Optional[Dict[str, float]] = None,
                               fitness_modifications: Optional[Dict[str, Any]] = None):
        """Save detailed report including incremental training info"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        incremental_report = {
            'incremental_training': {
                'previous_model_used': self.previous_model_path,
                'bounty_modifications': bounty_modifications or {},
                'fitness_modifications': fitness_modifications or {},
                'training_type': 'incremental' if self.previous_model_path else 'fresh'
            },
            'original_report': report
        }
        
        report_path = f"reports/incremental_training_report_{timestamp}.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(incremental_report, f, indent=2)
        
        print(f"Incremental training report saved to: {report_path}")
        return report_path

def main():
    """Example of incremental training with modified bounties"""
    
    # Base configuration
    config = {
        'population_size': 16,
        'games_per_evaluation': 3,
        'max_parallel_evaluations': 4,
        'tournament_size': 3,
        'mutation_rate': 0.1,
        'max_moves_per_game': 100,
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("=== V7P3R Incremental Training Demo ===")
    
    # Create incremental trainer
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    # Example: Modify bounty weights to emphasize different aspects
    bounty_modifications = {
        'tactical_weight': 5.0,        # Increase tactical play
        'king_safety_weight': 4.0,     # Increase king safety focus
        'center_control_weight': 1.5,  # Reduce center control emphasis
    }
    
    fitness_modifications = {
        'max_moves_per_game': 120,  # Allow longer games
        'mutation_rate': 0.08       # Reduce mutation rate for fine-tuning
    }
    
    # Run incremental training
    print("Starting incremental training with modified fitness function...")
    report = trainer.train_with_modified_fitness(
        num_generations=10,
        bounty_modifications=bounty_modifications,
        fitness_modifications=fitness_modifications
    )
    
    # Save detailed report
    trainer.save_incremental_report(report, bounty_modifications, fitness_modifications)
    
    print(f"Incremental training completed!")
    print(f"Best fitness achieved: {report['training_summary']['best_fitness_achieved']:.2f}")

if __name__ == "__main__":
    main()
