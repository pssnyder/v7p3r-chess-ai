# training_configs.py
"""
V7P3R Chess AI 2.0 - Training Configuration Presets
Different training configurations for various scenarios and testing phases.
"""

from v7p3r_genetic_trainer import GeneticConfig
import multiprocessing as mp


def get_initial_exploration_config():
    """
    Initial training config for exploring bounty system effectiveness.
    Fast iterations, good for seeing if heuristics are working.
    Expected runtime: 1-2 hours
    """
    return GeneticConfig(
        population_size=32,           # Smaller for faster iterations
        generations=50,               # Quick feedback loop
        mutation_rate=0.25,           # Higher exploration initially
        mutation_strength=0.03,       # Slightly stronger mutations
        crossover_rate=0.6,           # Moderate crossover
        elite_percentage=0.15,        # Keep more elites for stability
        tournament_size=4,            # Smaller tournaments
        games_per_individual=4,       # Good sample size for fitness
        max_moves_per_game=120,       # Shorter games to avoid draws
        parallel_workers=min(6, mp.cpu_count()),
        save_frequency=5,             # Save often for monitoring
        extended_bounty=True
    )


def get_development_config():
    """
    Development config for refining the AI after initial exploration.
    Balanced approach for continued improvement.
    Expected runtime: 3-4 hours
    """
    return GeneticConfig(
        population_size=64,           # Medium population
        generations=100,              # More generations
        mutation_rate=0.18,           # Reduced as AI improves
        mutation_strength=0.025,      # Finer adjustments
        crossover_rate=0.7,           # More crossover
        elite_percentage=0.12,        # Standard elitism
        tournament_size=5,            # Standard tournament
        games_per_individual=3,       # Standard games
        max_moves_per_game=150,       # Standard game length
        parallel_workers=min(8, mp.cpu_count()),
        save_frequency=10,            # Less frequent saves
        extended_bounty=True
    )


def get_production_config():
    """
    Full production training config for serious AI development.
    Expected runtime: 8-12 hours
    """
    return GeneticConfig(
        population_size=128,          # Full population
        generations=300,              # Extended training
        mutation_rate=0.15,           # Standard mutation
        mutation_strength=0.02,       # Fine-tuned mutations
        crossover_rate=0.7,           # Standard crossover
        elite_percentage=0.1,         # Standard elitism
        tournament_size=5,            # Standard tournament
        games_per_individual=3,       # Standard games
        max_moves_per_game=200,       # Full games
        parallel_workers=min(8, mp.cpu_count()),
        save_frequency=15,            # Periodic saves
        extended_bounty=True
    )


def get_quick_test_config():
    """
    Very quick config for testing changes and debugging.
    Expected runtime: 15-30 minutes
    """
    return GeneticConfig(
        population_size=16,           # Minimal population
        generations=20,               # Quick test
        mutation_rate=0.3,            # High exploration
        mutation_strength=0.04,       # Strong mutations
        crossover_rate=0.5,           # Less crossover
        elite_percentage=0.2,         # Keep best performers
        tournament_size=3,            # Small tournaments
        games_per_individual=2,       # Minimal games
        max_moves_per_game=80,        # Very short games
        parallel_workers=min(4, mp.cpu_count()),
        save_frequency=3,             # Frequent saves
        extended_bounty=True
    )


def get_bounty_tuning_config():
    """
    Special config for tuning bounty system effectiveness.
    Longer games, more detailed analysis.
    Expected runtime: 2-3 hours
    """
    return GeneticConfig(
        population_size=48,           # Medium population
        generations=75,               # Good sample size
        mutation_rate=0.2,            # Higher exploration
        mutation_strength=0.025,      # Moderate mutations
        crossover_rate=0.65,          # Balanced crossover
        elite_percentage=0.15,        # Keep good performers
        tournament_size=4,            # Smaller tournaments
        games_per_individual=5,       # More games for better fitness assessment
        max_moves_per_game=140,       # Longer for tactical development
        parallel_workers=min(6, mp.cpu_count()),
        save_frequency=5,             # Frequent monitoring
        extended_bounty=True
    )


# Configuration recommendations based on hardware
def get_config_for_hardware(cpu_cores: int, ram_gb: int) -> GeneticConfig:
    """Get optimal config based on available hardware"""
    
    if cpu_cores <= 4 and ram_gb <= 8:
        # Limited hardware
        return get_quick_test_config()
    elif cpu_cores <= 6 and ram_gb <= 16:
        # Moderate hardware
        config = get_initial_exploration_config()
        config.parallel_workers = min(4, cpu_cores)
        return config
    elif cpu_cores <= 8 and ram_gb <= 32:
        # Good hardware
        return get_development_config()
    else:
        # High-end hardware
        return get_production_config()


# Training phases recommendation
TRAINING_PHASES = {
    "phase_1_exploration": {
        "config": get_initial_exploration_config(),
        "description": "Initial exploration - test bounty system effectiveness",
        "goals": ["Verify bounty heuristics work", "Escape random play", "Identify best performers"],
        "duration": "1-2 hours",
        "success_metrics": ["Fitness > 20", "Consistent tactical improvements", "Bounty variance reduction"]
    },
    
    "phase_2_development": {
        "config": get_development_config(),
        "description": "Development phase - refine successful strategies",
        "goals": ["Improve tactical play", "Develop positional understanding", "Reduce blunders"],
        "duration": "3-4 hours", 
        "success_metrics": ["Fitness > 50", "Stable improvement", "Good game quality"]
    },
    
    "phase_3_production": {
        "config": get_production_config(),
        "description": "Production training - create strong AI",
        "goals": ["Achieve target strength", "Robust performance", "Tournament ready"],
        "duration": "8-12 hours",
        "success_metrics": ["Fitness > 100", "Consistent strong play", "Beat reference engines"]
    }
}
