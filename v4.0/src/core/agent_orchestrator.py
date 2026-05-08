"""
Agent Orchestrator
Coordinates multiple AI agents for chess decision-making
"""

import chess
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class AgentPriority(Enum):
    """Agent priority levels"""
    CRITICAL = 0  # Tablebase, forced mates
    HIGH = 1      # Opening book
    MEDIUM = 2    # Move ordering, themes
    LOW = 3       # Historical validation


@dataclass
class AgentMessage:
    """Message passed between agents and orchestrator"""
    sender: str
    receiver: str
    message_type: str  # 'move_request', 'evaluation', 'correction', etc.
    payload: Dict[str, Any]
    timestamp: float
    priority: AgentPriority


class AgentOrchestrator:
    """
    Multi-Agent Coordination System
    
    Manages communication and decision-making across specialized agents.
    Implements priority-based agent consultation with graceful fallback.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Agent Orchestrator
        
        Args:
            config_path: Path to agent_config.json
        """
        self.agents = {}
        self.agent_stats = {}
        self.fallback_enabled = True
        self.config = self._load_config(config_path)
        
        logger.info("Agent Orchestrator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load agent configuration"""
        if config_path:
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def register_agent(self, name: str, agent: Any, priority: AgentPriority):
        """
        Register an agent with the orchestrator
        
        Args:
            name: Agent identifier (e.g., 'themes', 'opening')
            agent: Agent instance
            priority: Agent priority level
        """
        self.agents[name] = {
            'instance': agent,
            'priority': priority,
            'enabled': True
        }
        self.agent_stats[name] = {
            'calls': 0,
            'successes': 0,
            'failures': 0,
            'avg_latency_ms': 0.0
        }
        logger.info(f"Registered agent: {name} (priority: {priority.name})")
    
    def get_move(
        self, 
        board: chess.Board, 
        time_limit: float = 3.0
    ) -> Optional[chess.Move]:
        """
        Get move from multi-agent system
        
        Consultation order:
        1. Opening agent (moves 1-15)
        2. Endgame agent (tablebase or mate detection)
        3. Main search with themes agent move ordering
        4. Corrector agent validation
        
        Args:
            board: Current chess position
            time_limit: Time budget for decision
            
        Returns:
            Best move or None
        """
        # Priority 1: Opening Agent
        if board.fullmove_number <= 15 and 'opening' in self.agents:
            opening_move = self._consult_agent('opening', 'get_opening_move', board)
            if opening_move:
                logger.info(f"Opening agent move: {opening_move}")
                return opening_move
        
        # Priority 2: Endgame Agent (perfect play)
        if len(board.piece_map()) <= 6 and 'endgame' in self.agents:
            endgame_move = self._consult_agent('endgame', 'get_endgame_move', board)
            if endgame_move and endgame_move.type in ['tablebase', 'forced_mate']:
                logger.info(f"Endgame agent move: {endgame_move.move} (type: {endgame_move.type})")
                return endgame_move.move
        
        # Priority 3: Mate Detection (any phase)
        if 'endgame' in self.agents:
            mate_move = self._consult_agent('endgame', 'find_mate', board, max_depth=3)
            if mate_move:
                logger.info(f"Forced mate: {mate_move.move} in {mate_move.mate_in}")
                return mate_move.move
        
        # Priority 4: Main Search with Themes Agent Move Ordering
        candidate_move = None
        
        if 'themes' in self.agents:
            # Get AI-ordered moves
            ranking = self._consult_agent('themes', 'rank_moves', board, time_limit)
            if ranking and ranking.ranked_moves:
                # For now, return top move
                # TODO: Integrate with full alpha-beta search
                candidate_move = ranking.ranked_moves[0]
                logger.info(f"Themes agent top move: {candidate_move}")
        
        if candidate_move is None:
            # Fallback to first legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                candidate_move = legal_moves[0]
        
        # Priority 5: Corrector Agent Validation
        if candidate_move and 'corrector' in self.agents:
            validation = self._consult_agent('corrector', 'validate_move', board, candidate_move)
            if validation and not validation.is_valid and validation.confidence > 0.7:
                logger.info(f"Corrector override: {candidate_move} → {validation.suggested_move}")
                candidate_move = validation.suggested_move
        
        return candidate_move
    
    def _consult_agent(self, agent_name: str, method_name: str, *args, **kwargs):
        """
        Consult an agent and track performance
        
        Args:
            agent_name: Name of agent to consult
            method_name: Method to call on agent
            *args, **kwargs: Arguments to pass to method
            
        Returns:
            Method result or None if failed
        """
        if agent_name not in self.agents:
            return None
        
        agent_info = self.agents[agent_name]
        if not agent_info['enabled']:
            return None
        
        agent = agent_info['instance']
        stats = self.agent_stats[agent_name]
        
        start_time = time.time()
        stats['calls'] += 1
        
        try:
            method = getattr(agent, method_name)
            result = method(*args, **kwargs)
            
            stats['successes'] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            # Update rolling average latency
            if stats['avg_latency_ms'] == 0:
                stats['avg_latency_ms'] = latency_ms
            else:
                stats['avg_latency_ms'] = 0.9 * stats['avg_latency_ms'] + 0.1 * latency_ms
            
            return result
            
        except Exception as e:
            stats['failures'] += 1
            logger.error(f"Agent {agent_name}.{method_name} failed: {e}")
            
            if self.fallback_enabled:
                logger.warning(f"Graceful fallback for agent {agent_name}")
            
            return None
    
    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all agents"""
        stats_summary = {}
        for name, stats in self.agent_stats.items():
            total_calls = stats['calls']
            if total_calls > 0:
                stats_summary[name] = {
                    'calls': total_calls,
                    'success_rate': stats['successes'] / total_calls,
                    'failure_rate': stats['failures'] / total_calls,
                    'avg_latency_ms': stats['avg_latency_ms']
                }
        return stats_summary
    
    def reset_stats(self):
        """Reset all agent statistics"""
        for name in self.agent_stats:
            self.agent_stats[name] = {
                'calls': 0,
                'successes': 0,
                'failures': 0,
                'avg_latency_ms': 0.0
            }
        logger.info("Agent statistics reset")


if __name__ == "__main__":
    # Quick test
    orchestrator = AgentOrchestrator()
    
    # Register agents (would be done with real agents)
    # orchestrator.register_agent('themes', themes_agent, AgentPriority.MEDIUM)
    
    board = chess.Board()
    move = orchestrator.get_move(board, time_limit=2.0)
    
    print(f"Selected move: {move}")
    print(f"Agent stats: {orchestrator.get_agent_stats()}")
